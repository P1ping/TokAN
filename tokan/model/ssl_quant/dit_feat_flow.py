import math
import logging
from typing import List, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from tokan.utils.mask import make_pad_mask

logger = logging.getLogger(__name__)


class DiTFeatFlow(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        spk_embed_dim: int,
        token_encoder: nn.Module,
        length_regulator: nn.Module,
        speech_decoder: nn.Module,
        duration_predictor: nn.Module = None,
        pitch_predictor: nn.Module = None,
        energy_predictor: nn.Module = None,
        teacher_pitch_predictor: nn.Module = None,
        normalize_spk_embed: bool = False,
        speech_feat_ratio: float = 1.0,
    ):
        super(DiTFeatFlow, self).__init__()

        self.token_encoder = token_encoder

        self.spk_embed_dim = spk_embed_dim
        self.normalize_spk_embed = normalize_spk_embed

        if duration_predictor is not None:
            self.spk_duration_proj = nn.Linear(spk_embed_dim, embed_dim)
        else:
            self.spk_duration_proj = None
        if (pitch_predictor is not None) or (energy_predictor is not None):
            self.spk_variance_proj = nn.Linear(spk_embed_dim, embed_dim)
        else:
            self.spk_variance_proj = None

        self.duration_predictor = duration_predictor
        self.length_regulator = length_regulator
        self.pitch_predictor = pitch_predictor
        self.energy_predictor = energy_predictor
        self.teacher_pitch_predictor = teacher_pitch_predictor

        self.speech_decoder = speech_decoder
        self.speech_feat_ratio = speech_feat_ratio  # Ratio of speech feature length to source feature length

        token_encoder_params = sum(p.numel() for p in self.token_encoder.parameters())
        speech_decoder_params = sum(p.numel() for p in self.speech_decoder.parameters())
        duration_predictor_params = (
            sum(p.numel() for p in self.duration_predictor.parameters()) if self.duration_predictor is not None else 0
        )
        logger.info(f"Token encoder parameters: {token_encoder_params/1e6:.2f}M")
        logger.info(f"Speech decoder parameters: {speech_decoder_params/1e6:.2f}M")
        logger.info(f"Duration predictor parameters: {duration_predictor_params/1e6:.2f}M")

    def forward(
        self,
        token_embed: torch.Tensor,
        token_embed_len: torch.Tensor,
        spk_embed: torch.Tensor,
        speech_feat: torch.Tensor,
        speech_feat_len: torch.Tensor,
        duration: torch.Tensor,
    ) -> List[torch.Tensor]:
        device = token_embed.device
        spk_embed = F.normalize(spk_embed, p=2, dim=-1) if self.normalize_spk_embed else spk_embed

        pitch, energy = None, None
        if self.teacher_pitch_predictor is not None:
            pitch, energy = self.prepare_variance_targets(speech_feat, speech_feat_len)

        # Token encoding
        B, T_token, D = token_embed.size()
        # token_embed, token_mask = self.encode_token(token_embed, token_embed_len)
        token_embed, token_mask = self.encode_token(token_embed, token_embed_len)

        # Duration prediction
        dp_loss = torch.tensor(0.0, device=device)
        if self.duration_predictor is not None:
            spk_embed_dp = self.spk_duration_proj(spk_embed)
            dp_input = token_embed.detach() + repeat(spk_embed_dp, "b d -> b t d", t=T_token)
            dp_loss = self.duration_predictor.compute_loss(dp_input, token_mask, duration)

        # NOTE: Here we quantize the scaled duration values while meeting the total duration
        cond, cond_len = self.length_regulator(token_embed, duration, token_mask, total_duration=speech_feat_len)
        cond_mask = ~make_pad_mask(cond_len, cond.size(1))

        if self.spk_variance_proj is not None:
            spk_embed_vp = self.spk_variance_proj(spk_embed)
            vp_input = cond.detach() + repeat(spk_embed_vp, "b d -> b t d", t=cond.size(1))

        pitch_loss = torch.tensor(0.0, device=device)
        if self.pitch_predictor is not None:
            pitch_loss, pitch_embed = self.pitch_predictor.compute_loss(vp_input, cond_mask, pitch)
            vp_input = vp_input + pitch_embed.detach()
            cond = cond + pitch_embed

        energy_loss = torch.tensor(0.0, device=device)
        if self.energy_predictor is not None:
            energy_loss, energy_embed = self.energy_predictor.compute_loss(vp_input, cond_mask, energy)
            cond = cond + energy_embed

        # Speech decoding
        B, T, D = speech_feat.shape
        speech_feat_mask = ~make_pad_mask(speech_feat_len, T)
        fm_loss = self.speech_decoder.compute_loss(speech_feat, speech_feat_mask, cond, spk_embed)

        return fm_loss, dp_loss, pitch_loss, energy_loss

    @torch.inference_mode()
    def inference(
        self,
        token_embed,
        token_embed_len,
        spk_embed,
        duration=None,
        total_duration=None,
        n_timesteps=32,
        full_cfg=0.0,
        cond_cfg=1.0,
        spk_cfg=1.0,
    ):
        """
        Args:
            token_embed (LongTensor): [B, T_token], text token ids.
            token_embed_len (LongTensor): [B], speech token lengths.
            spk_embed (FloatTensor): [B, D_spk], speaker embedding.
            duration (LongTensor, optional): [B, T_token], duration values of speech tokens. If None, use duration predictor.
            total_duration (LongTensor, optional): [B], total duration values of speech tokens. If None, no total duration constraint.
            n_timesteps (int): number of diffusion timesteps.
            full_cfg (float): full classifier-free guidance scale for unconditional sampling.
            cond_cfg (float): condition classifier-free guidance scale.
            spk_cfg (float): speaker classifier-free guidance scale.
        Returns:
            speech_feat (FloatTensor): [B, T_feat, D_feat], speech features.
            speech_feat_len (LongTensor): [B], speech feature lengths.
        """
        if self.normalize_spk_embed:
            spk_embed = F.normalize(spk_embed, p=2, dim=-1)

        B, T_token, D = token_embed.size()
        token_embed, token_mask = self.encode_token(token_embed, token_embed_len)

        if duration is None:
            assert self.duration_predictor is not None, "Duration predictor should be defined when duration is None."
            spk_embed_dp = self.spk_duration_proj(spk_embed)
            dp_input = token_embed.detach() + repeat(spk_embed_dp, "b d -> b t d", t=T_token)
            duration = self.duration_predictor(dp_input, token_mask, total_duration=total_duration)  # [B, T]

        cond, cond_len = self.length_regulator(token_embed, duration, token_mask, total_duration=duration.sum(dim=1))
        cond_mask = ~make_pad_mask(cond_len, cond.size(1))

        # Variance prediction and fusion
        if self.spk_variance_proj is not None:
            spk_embed_vp = self.spk_variance_proj(spk_embed)
            vp_input = cond + repeat(spk_embed_vp, "b d -> b t d", t=cond.size(1))

        if self.pitch_predictor is not None:
            pitch_pred, pitch_embed = self.pitch_predictor(vp_input, cond_mask, None)
            vp_input = vp_input + pitch_embed
            cond = cond + pitch_embed

        if self.energy_predictor is not None:
            energy_pred, energy_embed = self.energy_predictor(vp_input, cond_mask, None)
            cond = cond + energy_embed

        # Speech decoding
        speech_feat = self.speech_decoder.inference(
            cond_mask,
            cond,
            spk_embed,
            t_scheduler="cosine",
            n_timesteps=n_timesteps,
            full_cfg=full_cfg,
            cond_cfg=cond_cfg,
            spk_cfg=spk_cfg,
        )

        return speech_feat, cond_len

    def encode_token(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.token_encoder(
            tokens, token_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1
        )
        return encoder_out, encoder_mask.squeeze(1)

    # def encode_token(
    #     self,
    #     tokens: torch.Tensor,
    #     token_lengths: torch.LongTensor,
    # ):
    #     B, T, D = tokens.size()
    #     p = torch.arange(0, T, 1, dtype=tokens.dtype, device=tokens.device)  # [T]
    #     p = repeat(p, "t -> b t", b=B)  # [B, T]
    #     token_mask = ~make_pad_mask(token_lengths, T)
    #     attn_mask = token_mask.unsqueeze(1).expand(-1, T, -1)  # [B, T, T]
    #     tokens = self.token_encoder(tokens, p, attn_mask)
    #     return tokens, token_mask

    def prepare_variance_targets(self, fbank, variance_lengths):
        assert self.teacher_pitch_predictor is not None, "Teacher pitch predictor is required for variance targets."

        pitch = self.teacher_pitch_predictor(fbank.transpose(1, 2))  # [B, T]
        energy = torch.norm(fbank, dim=2)  # [B, T]

        # NOTE: Here we assume equal scales across samples, thus using the max length for interpolation
        pitch = F.interpolate(pitch.unsqueeze(1), size=variance_lengths.max(), mode="linear").squeeze(1)
        energy = F.interpolate(energy.unsqueeze(1), size=variance_lengths.max(), mode="linear").squeeze(1)

        return pitch, energy
