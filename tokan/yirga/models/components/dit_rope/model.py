# Copyright (c) 2025 TokAN Project
# Original implementation by Zhijun Liu
# Adapted for TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import torch
from torch import nn, Tensor, BoolTensor
from .layers import DiTBlock, ScalarEmbedder
from .rope import compute_r


class DiT(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        N_head: int,
        N_layer: int,
        P_dropout: float,
    ):
        super().__init__()
        self.t_encoder = ScalarEmbedder(256, D)
        self.blocks = nn.ModuleList([DiTBlock(D, D_hidden, N_head, P_dropout) for _ in range(N_layer)])
        self.C = D // N_head

    def forward(self, x: Tensor, p: Tensor, t: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            t (Tensor): [N, ...], the time tensor.
            mask (BoolTensor): [N, T, T], attention mask.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        t_emb = self.t_encoder.forward(t)
        r = compute_r(p, self.C)
        for block in self.blocks:
            x = block.forward(x, t_emb, r, mask)
        return x

    def get_attn(self, x: Tensor, p: Tensor, t: Tensor, mask: BoolTensor) -> Tensor:
        """Compute attention matrix.
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            t (Tensor): [N, ...], the time tensor.
            mask (BoolTensor): [N, T, T], attention mask.
        Returns:
            attn (Tensor): [N, ..., N_layer, H, T, T].
        """
        state = self.training
        attns = []
        with torch.inference_mode():
            self.eval()
            t_emb = self.t_encoder.forward(t)
            r = compute_r(p, self.C)
            for block in self.blocks:
                attn = block.get_attn(x, t_emb, r, mask)
                x = block.forward(x, t_emb, r, mask)
                attns.append(attn)
        self.train(state)
        return torch.stack(attns, dim=-4)
