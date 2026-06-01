from typing import Dict, Optional, Callable, List, Generator, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from tokan.utils.common import IGNORE_ID
from tokan.transformer.label_smoothing_loss import LabelSmoothingLoss
from tokan.utils.common import th_accuracy
from tokan.utils.file_utils import logging


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        lm_input_dim: int,
        lm_output_dim: int,
        speech_vocab_size: int,
        speech_embed_dim: int,
        token_encoder: torch.nn.Module,
        ctc_vocab_size: int,
        ctc_target_key: str,
        lm: torch.nn.Module,
        sampling: Callable,
        share_speech_embedding: bool = True,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        src_embed_dim: int = 0,
        ctc_loss_weight: float = 1.0,
        mask_in_input: bool = False,
    ):
        super().__init__()
        self.lm_input_dim = lm_input_dim
        self.speech_vocab_size = speech_vocab_size
        self.ctc_loss_weight = ctc_loss_weight
        self.ctc_vocab_size = ctc_vocab_size
        self.ctc_target_key = ctc_target_key

        self.source_speech_embedding = torch.nn.Embedding(
            speech_vocab_size + 1 if mask_in_input else speech_vocab_size, speech_embed_dim
        )
        self.token_encoder = token_encoder
        self.token_encoder_output_proj = nn.Linear(token_encoder.output_size(), lm_input_dim)
        if ctc_vocab_size > 0:
            self.ctc_proj = nn.Linear(token_encoder.output_size(), ctc_vocab_size)
        else:
            self.ctc_proj = None

        self.sos_eos = 0
        self.task_id = 1
        self.special_embedding = torch.nn.Embedding(2, lm_input_dim)
        self.lm = lm
        self.lm_output_proj = nn.Linear(lm_output_dim, speech_vocab_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_vocab_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if share_speech_embedding:
            assert speech_embed_dim == lm_input_dim, "For shared speech embedding, dimensions must match."
            self.target_speech_embedding = self.source_speech_embedding
        else:
            self.target_speech_embedding = torch.nn.Embedding(speech_vocab_size, lm_input_dim)
        if src_embed_dim > 0:
            self.src_embed_proj = torch.nn.Linear(src_embed_dim, lm_input_dim)
            self.use_src_embedding = True
        else:
            self.src_embed_proj = None
            self.use_src_embedding = False

        self.sampling = sampling

        encoder_params = sum(p.numel() for p in self.token_encoder.parameters() if p.requires_grad)
        lm_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            f"TransformerLM total parameters: {total_params} "
            f"= encoder: {encoder_params} + lm: {lm_params} + others: {total_params - encoder_params - lm_params}"
        )

    def encode(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lengths: torch.Tensor,
    ):
        # encoder_out, encoder_mask = self.token_encoder(
        #     speech_tokens, speech_token_lengths, decoding_chunk_size=-1, num_decoding_left_chunks=-1
        # )
        # encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # encoder_out = self.token_encoder_output_proj(encoder_out)
        # ctc_logits = self.ctc_proj(encoder_out) if self.ctc_proj is not None else None
        # return encoder_out, encoder_out_lens, ctc_logits

        pos, encoder_mask = self.make_position_and_mask(
            speech_token_lengths, cond_lengths=torch.zeros_like(speech_token_lengths)
        )
        encoder_out = self.token_encoder(speech_tokens, pos, encoder_mask)
        ctc_logits = self.ctc_proj(encoder_out) if self.ctc_proj is not None else None
        encoder_out = self.token_encoder_output_proj(encoder_out)

        return encoder_out, speech_token_lengths, ctc_logits

    def prepare_lm_input(
        self,
        sos_eos_emb: torch.Tensor,
        src_embedding: Optional[torch.Tensor],
        src_tokens: torch.Tensor,
        src_token_len: torch.Tensor,
        task_id_emb: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_token_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build [<sos>, (<src_embed>), src, <task>, tgt] using scatter."""
        bsz, _, feat_dim = src_tokens.size()
        device = src_tokens.device
        dtype = src_tokens.dtype

        prefix_len = 1 + (1 if self.use_src_embedding else 0)
        task_id_idx = prefix_len + src_token_len
        tgt_start_idx = task_id_idx + 1
        total_lengths = tgt_start_idx + tgt_token_len

        total_max_len = total_lengths.max().item()
        lm_input = torch.zeros((bsz, total_max_len, feat_dim), device=device, dtype=dtype)

        lm_input[:, 0:1] = sos_eos_emb
        src_start = 1
        if self.use_src_embedding:
            lm_input[:, 1:2] = src_embedding
            src_start = 2

        _, src_max_len, _ = src_tokens.shape
        src_write_indices = torch.arange(src_max_len, device=device).unsqueeze(0).expand(bsz, -1) + src_start
        src_indices_expanded = src_write_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        lm_input.scatter_(1, src_indices_expanded.long(), src_tokens)

        task_indices_expanded = task_id_idx.view(-1, 1, 1).expand(-1, 1, feat_dim)
        lm_input.scatter_(1, task_indices_expanded.long(), task_id_emb.expand(bsz, -1, -1).to(dtype))

        tgt_max_len = tgt_tokens.size(1)
        tgt_grid = torch.arange(tgt_max_len, device=device).unsqueeze(0)
        tgt_write_indices = tgt_start_idx.unsqueeze(1) + tgt_grid
        safe_tgt_write_indices = tgt_write_indices.clamp(max=total_max_len - 1)
        tgt_indices_expanded = safe_tgt_write_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
        lm_input.scatter_(1, tgt_indices_expanded.long(), tgt_tokens.to(dtype))

        total_grid = torch.arange(total_max_len, device=device).unsqueeze(0).expand(bsz, -1)
        valid_total_mask = total_grid < total_lengths.unsqueeze(1)
        lm_input = lm_input.masked_fill(~valid_total_mask.unsqueeze(-1), IGNORE_ID)

        return lm_input, total_lengths, tgt_start_idx, task_id_idx, safe_tgt_write_indices, valid_total_mask

    def prepare_lm_target(
        self,
        task_id_idx: torch.Tensor,
        tgt_write_indices: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_token_len: torch.Tensor,
        total_len: int,
    ) -> torch.Tensor:
        """Build next-token targets for AR training.

        Input layout:   [<sos>, (<src_embed>), <src_seq>, <task_id>, tgt[0], tgt[1], ..., tgt[L-1]]
        Target layout:  [IGNORE...,            IGNORE,    tgt[0],    tgt[1], tgt[2], ..., tgt[L-1], <eos>]

        We scatter L+1 targets in one shot using unified index and value arrays.
        """
        bsz, tgt_max_len = tgt_tokens.shape
        device = tgt_tokens.device

        lm_target = torch.full((bsz, total_len), IGNORE_ID, dtype=torch.long, device=device)

        # Unified scatter indices: [task_id_pos, tgt_pos[0], ..., tgt_pos[tgt_max_len-1]]
        # shape: (bsz, 1 + tgt_max_len)
        all_indices = torch.cat([task_id_idx.unsqueeze(1), tgt_write_indices], dim=1)

        # Start with [tgt[0], ..., tgt[tgt_max_len-1], <eos>] and fix EOS placement:
        # so we scatter <eos> at the correct column per sample.
        eos_col = torch.full((bsz, 1), self.speech_vocab_size, dtype=torch.long, device=device)
        all_targets = torch.cat([tgt_tokens, eos_col], dim=1)  # (bsz, 1 + tgt_max_len)
        all_targets.scatter_(1, tgt_token_len.unsqueeze(1).clamp(max=tgt_max_len).long(), eos_col)

        # Mask positions beyond <eos> (k > tgt_token_len) to IGNORE_ID so they do not contribute to the loss.
        k = torch.arange(1 + tgt_max_len, device=device).unsqueeze(0)  # (1, 1+tgt_max_len)
        all_targets.masked_fill_(k > tgt_token_len.unsqueeze(1), IGNORE_ID)

        lm_target.scatter_(1, all_indices, all_targets)

        return lm_target

    def make_position_and_mask(
        self,
        total_lengths: torch.Tensor,
        cond_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate position ids -- regarding the entire sequence as a whole
        max_len = total_lengths.max().item()
        base_pos = torch.arange(0, max_len, 1, dtype=torch.long, device=total_lengths.device)
        pos = repeat(base_pos, "t -> b t", b=total_lengths.shape[0])

        key_is_valid = pos < total_lengths.unsqueeze(1)
        query_is_valid = key_is_valid

        # Make a causal mask
        key_pos = pos.unsqueeze(1)
        query_pos = pos.unsqueeze(2)
        causal_mask = key_pos <= query_pos
        attn_mask = causal_mask
        attn_mask = attn_mask & query_is_valid.unsqueeze(2) & key_is_valid.unsqueeze(1)

        return pos, attn_mask

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        src_tokens = batch["src_token"].to(device)
        src_token_len = batch["src_token_len"].to(device)
        tgt_tokens = batch["tgt_token"].to(device)
        tgt_token_len = batch["tgt_token_len"].to(device)

        if self.ctc_loss_weight > 0:
            aux_tokens = batch[self.ctc_target_key].to(device)
            aux_len = batch[f"{self.ctc_target_key}_len"].to(device)
        else:
            aux_tokens = None
            aux_len = None
        if self.use_src_embedding:
            src_embedding = batch["src_embedding"].to(device)
        else:
            src_embedding = None

        src_tokens_embedded = self.source_speech_embedding(src_tokens)
        encoded_source, encoded_source_len, ctc_logits = self.encode(src_tokens_embedded, src_token_len)

        if self.use_src_embedding:
            src_embedding = F.normalize(src_embedding, dim=1)
            src_embedding = self.src_embed_proj(src_embedding)
            src_embedding = src_embedding.unsqueeze(1)

        sos_eos_emb = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)
        tgt_tokens_embedded = self.target_speech_embedding(tgt_tokens)

        lm_input, lm_input_len, tgt_start_idx, task_id_idx, tgt_write_indices, _ = self.prepare_lm_input(
            sos_eos_emb,
            src_embedding,
            encoded_source,
            encoded_source_len,
            task_id_emb,
            tgt_tokens_embedded,
            tgt_token_len,
        )

        lm_target = self.prepare_lm_target(
            task_id_idx=task_id_idx,
            tgt_write_indices=tgt_write_indices,
            tgt_tokens=tgt_tokens,
            tgt_token_len=tgt_token_len,
            total_len=lm_input.size(1),
        )

        position, attn_mask = self.make_position_and_mask(lm_input_len, tgt_start_idx)
        lm_output = self.lm(lm_input, position, attn_mask)
        logits = self.lm_output_proj(lm_output)
        ce_loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_vocab_size + 1), lm_target, ignore_label=IGNORE_ID)

        if self.ctc_loss_weight > 0:
            ctc_logp = ctc_logits.log_softmax(dim=-1)
            ctc_loss = F.ctc_loss(
                ctc_logp.transpose(0, 1),
                aux_tokens,
                encoded_source_len,
                aux_len,
                blank=0,
                zero_infinity=True,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=device)

        loss = ce_loss + ctc_loss * self.ctc_loss_weight

        return {
            "loss": loss,
            "acc": acc,
            "ctc_loss": ctc_loss,
            "ce_loss": ce_loss,
            "batch_size": torch.tensor(src_tokens.size(0), device=device),
        }

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_vocab_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    def _gather_kv_cache(
        self, caches: List[Optional[torch.Tensor]], indices: torch.Tensor
    ) -> List[Optional[torch.Tensor]]:
        gathered = []
        for cache in caches:
            if cache is None:
                gathered.append(None)
            else:
                gathered.append(cache[indices])
        return gathered

    @torch.inference_mode()
    def beam_search_decode(
        self,
        cond_input: torch.Tensor,
        cond_len: int,
        beam_size: int,
        min_len: int,
        max_len: int,
    ) -> List[int]:
        device = cond_input.device
        eos_id = self.speech_vocab_size
        num_layers = len(self.lm.blocks)

        # 1. Process conditioning sequence once
        k_caches = [None] * num_layers
        v_caches = [None] * num_layers
        cond_pos = torch.arange(cond_len, device=device, dtype=torch.long).unsqueeze(0)
        cond_mask = torch.ones((1, cond_len, cond_len), device=device, dtype=torch.bool)
        cond_out, k_caches, v_caches = self.lm.forward_incremental(cond_input, cond_pos, cond_mask, k_caches, v_caches)

        # 2. Initialize beam search from first log-probs
        log_probs = self.lm_output_proj(cond_out[:, -1]).log_softmax(dim=-1)  # (1, V)
        beam_scores = torch.zeros(1, device=device)
        beam_tokens_history = [[]]
        completed_beams = []

        for step in range(max_len):
            if step < min_len:
                log_probs[:, eos_id] = -float("inf")

            next_scores = beam_scores.unsqueeze(1) + log_probs  # (B, V)
            next_scores_flat = next_scores.reshape(-1)
            n_candidates = min(2 * beam_size, next_scores_flat.size(0))
            top_scores, top_indices = torch.topk(next_scores_flat, k=n_candidates)

            vocab_size = log_probs.shape[-1]
            prev_beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            max_active_beams = beam_size - len(completed_beams)
            if max_active_beams <= 0:
                break

            next_beam_scores = []
            next_beam_tokens_history = []
            next_beam_parent_indices = []
            next_tokens = []

            for i in range(n_candidates):
                if len(next_tokens) == max_active_beams:
                    break

                score = top_scores[i].item()
                token_id = token_indices[i].item()
                beam_idx = prev_beam_indices[i].item()
                prev_history = beam_tokens_history[beam_idx]

                if token_id == eos_id:
                    final_score = score / (len(prev_history) + 1e-6)
                    completed_beams.append((final_score, prev_history))
                else:
                    next_beam_scores.append(score)
                    next_beam_tokens_history.append(prev_history + [token_id])
                    next_beam_parent_indices.append(beam_idx)
                    next_tokens.append(token_id)

            if len(next_tokens) == 0:
                break

            # Reorder caches for surviving parent beams
            parent_indices = torch.tensor(next_beam_parent_indices, dtype=torch.long, device=device)
            k_caches = self._gather_kv_cache(k_caches, parent_indices)
            v_caches = self._gather_kv_cache(v_caches, parent_indices)

            # Step forward with one new token per beam
            B = len(next_tokens)
            next_tokens_tensor = torch.tensor(next_tokens, device=device)
            next_input = self.target_speech_embedding.weight[next_tokens_tensor].unsqueeze(1)  # (B, 1, D)
            total_len = cond_len + step + 1
            pos = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            attn_mask = torch.ones((B, 1, total_len), device=device, dtype=torch.bool)
            step_out, k_caches, v_caches = self.lm.forward_incremental(next_input, pos, attn_mask, k_caches, v_caches)
            log_probs = self.lm_output_proj(step_out[:, -1]).log_softmax(dim=-1)

            beam_scores = torch.tensor(next_beam_scores, device=device)
            beam_tokens_history = next_beam_tokens_history

        # 3. Finalize: fold in any beams that hit max_len without EOS
        for i, history in enumerate(beam_tokens_history):
            if len(history) > 0:
                final_score = beam_scores[i].item() / len(history)
                completed_beams.append((final_score, history))

        if not completed_beams:
            return []
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        return completed_beams[0][1]

    @torch.inference_mode()
    def inference(
        self,
        src_tokens: torch.Tensor,
        src_token_len: torch.Tensor,
        prompt_src_tokens: torch.Tensor = None,
        prompt_src_token_len: torch.Tensor = None,
        prompt_tgt_tokens: torch.Tensor = None,
        prompt_tgt_token_len: torch.Tensor = None,
        src_embedding: torch.Tensor = None,
        sampling: int = 25,
        beam_size: int = 0,
        max_token_ratio: float = 5,
        min_token_ratio: float = 0.2,
    ) -> Generator[torch.Tensor, None, None]:
        device = src_tokens.device

        if prompt_src_tokens is not None and prompt_src_tokens.size(1) > 0:
            src_tokens = torch.concat([prompt_src_tokens, src_tokens], dim=1)
            src_token_len = src_token_len + prompt_src_token_len

        src_tokens_embedded = self.source_speech_embedding(src_tokens)
        encoded_source, encoded_source_len, _ = self.encode(src_tokens_embedded, src_token_len)

        if self.use_src_embedding:
            src_embedding = F.normalize(src_embedding, dim=1)
            src_embedding = self.src_embed_proj(src_embedding)
            src_embedding = src_embedding.unsqueeze(1)
        else:
            src_embedding = None

        sos_eos_emb = self.special_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.special_embedding.weight[self.task_id].reshape(1, 1, -1)

        if prompt_tgt_tokens is not None and prompt_tgt_token_len is not None and prompt_tgt_token_len.item() > 0:
            prompt_tgt_emb = self.target_speech_embedding(prompt_tgt_tokens)
            prompt_tgt_len = prompt_tgt_token_len.to(device)
        else:
            prompt_tgt_emb = torch.zeros(1, 0, self.lm_input_dim, dtype=encoded_source.dtype, device=device)
            prompt_tgt_len = torch.tensor([0], device=device, dtype=torch.long)

        cond_input, cond_total_len, _, _, _, _ = self.prepare_lm_input(
            sos_eos_emb,
            src_embedding,
            encoded_source,
            encoded_source_len,
            task_id_emb,
            prompt_tgt_emb,
            prompt_tgt_len,
        )
        cond_len = cond_total_len.item()

        input_len = src_token_len
        if prompt_src_tokens is not None:
            input_len = input_len - prompt_src_token_len
        input_len_scalar = input_len.item()
        min_len = int(input_len_scalar * min_token_ratio)
        max_len = int(input_len_scalar * max_token_ratio)

        if beam_size >= 1:
            out_tokens = self.beam_search_decode(
                cond_input=cond_input,
                cond_len=cond_len,
                beam_size=beam_size,
                min_len=min_len,
                max_len=max_len,
            )
            for token_id in out_tokens:
                yield token_id
            return

        num_layers = len(self.lm.blocks)
        k_caches = [None] * num_layers
        v_caches = [None] * num_layers

        cond_pos = torch.arange(cond_len, device=device, dtype=torch.long).unsqueeze(0)
        cond_mask = torch.ones((1, cond_len, cond_len), device=device, dtype=torch.bool)
        cond_out, k_caches, v_caches = self.lm.forward_incremental(cond_input, cond_pos, cond_mask, k_caches, v_caches)

        out_tokens = []
        logp = self.lm_output_proj(cond_out[:, -1]).log_softmax(dim=-1)

        for i in range(max_len):
            if i < min_len:
                logp[:, self.speech_vocab_size] = -float("inf")

            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=(i < min_len)).item()
            if top_ids == self.speech_vocab_size:
                break

            yield top_ids
            out_tokens.append(top_ids)

            step_input = self.target_speech_embedding.weight[top_ids].reshape(1, 1, -1)
            total_len = cond_len + i + 1
            step_pos = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)
            step_mask = torch.ones((1, 1, total_len), device=device, dtype=torch.bool)
            step_out, k_caches, v_caches = self.lm.forward_incremental(
                step_input,
                step_pos,
                step_mask,
                k_caches,
                v_caches,
            )
            logp = self.lm_output_proj(step_out[:, -1]).log_softmax(dim=-1)
