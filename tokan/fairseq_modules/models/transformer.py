# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import logging
from typing import Optional, Dict, Tuple

import torch
from torch import nn
from torch.nn.functional import layer_norm
from einops import rearrange

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.models.transformer.transformer_encoder import TransformerEncoderBase
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.models.transformer.transformer_legacy import TransformerModel
from fairseq.models.transformer.transformer_legacy import (
    base_architecture as base_architecture_legacy,
)

logger = logging.getLogger(__name__)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, feat_dim: int, condition_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(feat_dim))
        self.bias = nn.Parameter(torch.zeros(feat_dim)) if bias else None
        self.modulation_net = nn.Sequential(
            nn.Linear(condition_dim, feat_dim // 2), nn.SiLU(), nn.Linear(feat_dim // 2, feat_dim * 2)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [T, B, D], Input sequential features.
            cond (Tensor): [B, D'], Conditional features.
        Returns:
            Tensor: [T, B, D] Normalized and adapted features.
        """
        normalized = layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

        scale_shift = self.modulation_net(cond)  # [B, D*2]
        scale_shift = rearrange(scale_shift, "b d -> 1 b d")
        scale, shift = scale_shift.chunk(2, dim=-1)

        return normalized * (1 + scale) + shift


class TransformerEncoderLayerAdaLN(TransformerEncoderLayerBase):
    def __init__(self, cfg, condition_dim):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)

        # Replace standard LayerNorms with AdaLNs
        self.self_attn_layer_norm = AdaptiveLayerNorm(feat_dim=self.embed_dim, condition_dim=condition_dim)

        self.dropout_module = FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout or cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = AdaptiveLayerNorm(feat_dim=self.embed_dim, condition_dim=condition_dim)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x, condition)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x, condition)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x, condition)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x, condition)

        return x


class TransformerEncoderAdaLN(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)
        self.condition_dim = getattr(cfg, "condition_dim", None)
        self.condition_embeddings = self.build_condition_embeddings(cfg)

    def build_encoder_layer(self, cfg):
        condition_dim = getattr(cfg, "condition_dim", None)
        if condition_dim is not None:
            layer = TransformerEncoderLayerAdaLN(cfg, condition_dim=condition_dim)
        else:
            layer = TransformerEncoderLayerBase(cfg)

        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)

        return layer

    def build_condition_embeddings(self, cfg):
        if cfg.num_conditions is not None:
            assert cfg.condition_dim is not None
            condition_embeddings = nn.Embedding(cfg.num_conditions, cfg.condition_dim)
        else:
            condition_embeddings = None
        return condition_embeddings

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        condition_labels: Optional[torch.Tensor] = None,
        condition_embeds: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            condition_labels (torch.Tensor, optional): condition label tensor of shape
                shape `(batch)`
            condition_embeds (torch.Tensor, optional): condition embedding tensor of shape
                shape `(batch, condition_dim)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = torch.tensor(src_tokens.device.type == "xla") or encoder_padding_mask.any()
        # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
        if torch.jit.is_scripting():
            has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

        if condition_embeds is not None:
            cond = condition_embeds
        elif self.condition_embeddings is not None:
            cond = self.condition_embeddings(condition_labels)
        else:
            cond = None

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            if self.condition_dim is not None:
                lr = layer(
                    x,
                    encoder_padding_mask=encoder_padding_mask if has_pads else None,
                    condition=cond,
                )
            else:
                lr = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, torch.Tensor]):
        encoder_input = {k: v for k, v in net_input.items() if k not in ["prev_output_tokens", "src_wavs"]}
        return self.forward(**encoder_input)


@register_model("speech_token_transformer")
class SpeechTokenTransformer(TransformerModel):
    """
    Transformer model which has the Whisper as the front-end encoder.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.condition_dim = getattr(cfg, "condition_dim", None)

        self.ctc_projection = (
            nn.Linear(cfg.encoder_embed_dim, cfg.num_ctc_classes) if cfg.num_ctc_classes is not None else None
        )

        if getattr(cfg, "pretrained_checkpoint", None):
            self.load_pretrained_checkpoint(cfg.pretrained_checkpoint)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderAdaLN(cfg, src_dict, embed_tokens)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        condition_labels: Optional[torch.LongTensor] = None,
        condition_embeds: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_wavs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            condition_labels=condition_labels,
            condition_embeds=condition_embeds,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        if self.ctc_projection is not None:
            ctc_out = self.ctc_projection(encoder_out["encoder_out"][0])  # T x B x V
            ctc_logp = ctc_out.log_softmax(dim=-1)  # T x B x V
            decoder_out[1]["ctc_lprobs"] = ctc_logp
        return decoder_out

    def get_ctc_output(self, net_output, sample):
        # CTC only supports full-precision computation
        ctc_lprobs = net_output[1]["ctc_lprobs"].float()  # T x B x V
        ctc_lens = sample["net_input"]["src_lengths"]
        return ctc_lprobs, ctc_lens

    def get_ctc_target(self, sample):
        ctc_tgt = sample["aux_texts"]
        ctc_tgt_lens = sample["aux_lengths"]
        return ctc_tgt, ctc_tgt_lens

    def load_pretrained_checkpoint(self, checkpoint_path):
        """
        Load a pretrained checkpoint while handling vocabulary size differences.

        Args:
            checkpoint_path (str): Path to the checkpoint file

        Raises:
            ValueError: If vocabulary size mismatch is inconsistent
            FileNotFoundError: If checkpoint file doesn't exist
        """
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Make sure to load fine-tuned model later.")
            return

        # Handle input and output embeddings
        encoder_token_increment = self._update_embeddings(
            current_embeds=self.encoder.embed_tokens.weight.data,
            pretrained_embeds=state_dict["encoder.embed_tokens.weight"],
            state_dict=state_dict,
            key="encoder.embed_tokens.weight",
        )
        decoder_token_increment = self._update_embeddings(
            current_embeds=self.decoder.embed_tokens.weight.data,
            pretrained_embeds=state_dict["decoder.embed_tokens.weight"],
            state_dict=state_dict,
            key="decoder.embed_tokens.weight",
        )
        # Update decoder embeddings if shared with encoder
        if decoder_token_increment != 0:
            state_dict = self._update_decoder_proj(state_dict, decoder_token_increment)

        state_dict = self._update_encoder_adaln(state_dict)

        # Load the modified state dict
        self.load_state_dict(state_dict, strict=True)

    def _update_embeddings(self, current_embeds, pretrained_embeds, state_dict, key):
        """
        Update embedding weights while handling vocabulary size differences.

        Args:
            current_embeds: Current model's embedding weights
            pretrained_embeds: Pretrained model's embedding weights
            state_dict: State dict to update
            key: Key in state dict for these embeddings

        Returns:
            int: Number of new tokens added (token_increment)
        """
        token_increment = current_embeds.size(0) - pretrained_embeds.size(0)
        if token_increment > 0:
            current_embeds[:-token_increment] = pretrained_embeds
            state_dict[key] = current_embeds
            logger.warning(f"Last {token_increment} tokens in {key} are randomly initialized")
        elif token_increment < 0:
            current_embeds = pretrained_embeds[:token_increment]
            state_dict[key] = current_embeds
            logger.warning(f"Removed {-token_increment} tokens from {key}")

        return token_increment

    def _update_decoder_proj(self, state_dict, token_increment):
        # Handle output projection
        out_proj = self.decoder.output_projection.weight.data
        pretrained_proj = state_dict["decoder.output_projection.weight"]

        if out_proj.size(0) - pretrained_proj.size(0) != token_increment:
            raise ValueError(
                f"Inconsistent token increment: {token_increment} vs "
                f"{out_proj.size(0) - pretrained_proj.size(0)} in output projection"
            )

        if token_increment > 0:
            out_proj[:-token_increment] = pretrained_proj
            state_dict["decoder.output_projection.weight"] = out_proj
            logger.warning(f"Last {token_increment} tokens in output projection are randomly initialized")
        elif token_increment < 0:
            out_proj = pretrained_proj[:token_increment]
            state_dict["decoder.output_projection.weight"] = out_proj
            logger.warning(f"Removed {-token_increment} tokens from output projection")

        return state_dict

    def _update_encoder_adaln(self, state_dict):
        for i, layer in enumerate(self.encoder.layers):
            if isinstance(layer, TransformerEncoderLayerAdaLN):
                weight_key_0 = f"encoder.layers.{i}.self_attn_layer_norm.modulation_net.0.weight"
                bias_key_0 = f"encoder.layers.{i}.self_attn_layer_norm.modulation_net.0.bias"
                weight_key_2 = f"encoder.layers.{i}.self_attn_layer_norm.modulation_net.2.weight"
                bias_key_2 = f"encoder.layers.{i}.self_attn_layer_norm.modulation_net.2.bias"

                if weight_key_0 not in state_dict:
                    state_dict[weight_key_0] = layer.self_attn_layer_norm.modulation_net[0].weight
                if bias_key_0 not in state_dict:
                    state_dict[bias_key_0] = layer.self_attn_layer_norm.modulation_net[0].bias
                if weight_key_2 not in state_dict:
                    state_dict[weight_key_2] = layer.self_attn_layer_norm.modulation_net[2].weight
                if bias_key_2 not in state_dict:
                    state_dict[bias_key_2] = layer.self_attn_layer_norm.modulation_net[2].bias

                # Update final_layer_norm
                final_weight_key_0 = f"encoder.layers.{i}.final_layer_norm.modulation_net.0.weight"
                final_bias_key_0 = f"encoder.layers.{i}.final_layer_norm.modulation_net.0.bias"
                final_weight_key_2 = f"encoder.layers.{i}.final_layer_norm.modulation_net.2.weight"
                final_bias_key_2 = f"encoder.layers.{i}.final_layer_norm.modulation_net.2.bias"

                if final_weight_key_0 not in state_dict:
                    state_dict[final_weight_key_0] = layer.final_layer_norm.modulation_net[0].weight
                if final_bias_key_0 not in state_dict:
                    state_dict[final_bias_key_0] = layer.final_layer_norm.modulation_net[0].bias
                if final_weight_key_2 not in state_dict:
                    state_dict[final_weight_key_2] = layer.final_layer_norm.modulation_net[2].weight
                if final_bias_key_2 not in state_dict:
                    state_dict[final_bias_key_2] = layer.final_layer_norm.modulation_net[2].bias

        # Handle condition embeddings
        if self.encoder.condition_embeddings is not None:
            condition_embeddings_key = "encoder.condition_embeddings.weight"
            if condition_embeddings_key not in state_dict:
                state_dict[condition_embeddings_key] = self.encoder.condition_embeddings.weight.data
                logger.warning("Encoder conditional embeddings are randomly initialized")

        return state_dict

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--pretrained-checkpoint",
            type=str,
            required=False,
        )
        parser.add_argument(
            "--num-conditions",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--condition-dim",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--num-ctc-classes",
            type=int,
            default=None,
        )
        gen_parser_from_dataclass(parser, TransformerConfig(), delete_default=True, with_prefix="")


@register_model_architecture("speech_token_transformer", "transformer_base++")
def basepp_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 5)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 5)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_architecture_legacy(args)
