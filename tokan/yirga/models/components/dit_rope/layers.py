# Copyright (c) 2025 TokAN Project
# Original implementation by Zhijun Liu
# Adapted for TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import math

import torch
from torch import nn, Tensor, BoolTensor
from torch.nn.functional import layer_norm, gelu
from .rope import RoPESelfAttention


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """
    Args:
        x (Tensor): [N, ..., T, D].
        shift (Tensor): [N, ..., D].
        scale-1.0 (Tensor): [N, ..., D].
    """
    return x * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, D: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """[..., D] -> [..., D]"""
        return layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        P_dropout: float,
        bias: bool = True,
    ):
        """
        Args:
            D (int): Input feature dimension.
            D_hidden (int): Hidden unit dimension.
            P_dropout (float): dropout value for first Linear Layer.
            bias (bool): If linear layers should have bias.
            d_cond (int, optional): The channels of conditional tensor.
        """
        super().__init__()
        self.w_1 = nn.Linear(D, D_hidden, bias=bias)
        self.drop_1 = nn.Dropout(P_dropout)
        self.w_2 = nn.Linear(D_hidden, D, bias=bias)
        self.drop_2 = nn.Dropout(P_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [..., D].
        Returns:
            y (Tensor): [..., D].
        """
        x = self.w_1(x)
        try:
            x = gelu(x, approximate="tanh")
        except TypeError:
            x = gelu(x)
        x = self.drop_1(x)
        x = self.w_2(x)
        return self.drop_2(x)


class DiTBlock(nn.Module):
    def __init__(self, D: int, D_hidden: int, N_head: int, P_dropout: float):
        super().__init__()
        self.attn_norm = LayerNorm(D)
        self.attn = RoPESelfAttention(D, N_head, P_dropout=P_dropout)
        self.ffn_norm = LayerNorm(D)
        self.ffn = FeedForwardModule(D, D_hidden, P_dropout, bias=True)
        self.AdaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D, bias=True))

    def forward(self, x: Tensor, c: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            c (Tensor): [N, ..., D], conditioning vector for AdaLN.
            r (Tensor): [N, ..., T, C//2], rotation in RoPE.
            mask (BoolTensor): [N, T, T], attention mask, True for valid positions.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.AdaLN_modulation(
            c
        ).chunk(6, dim=-1)
        # [N, ..., D]
        x1 = self.attn_norm.forward(x)
        x2 = modulate(x1, shift_msa, scale_msa)
        x3, _, _ = self.attn.forward(x2, r, mask)
        x4 = gate_msa.unsqueeze(-2) * x3
        x = x + x4
        x5 = self.ffn_norm.forward(x)
        x6 = modulate(x5, shift_mlp, scale_mlp)
        x7 = self.ffn.forward(x6)
        x8 = gate_mlp.unsqueeze(-2) * x7
        x = x + x8
        return x

    def get_attn(self, x: Tensor, c: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """A hack to obtain the attention matrix.
        Args:
            x (Tensor): [N, ..., T, D].
            r (ComplexFloatTensor): [N, ..., T, C // 2].
        Returns:
            attn (Tensor): [N, ..., H, T, T] the attention weights.
        """
        state = self.training
        with torch.no_grad():
            self.eval()
            (shift_msa, scale_msa, _, _, _, _) = self.AdaLN_modulation.forward(c).chunk(6, dim=-1)
            x1 = self.attn_norm.forward(x)
            x2 = modulate(x1, shift_msa, scale_msa)
            attn = self.attn.get_attn(x2, r, mask)
        self.train(state)
        return attn


class FinalLinear(nn.Module):
    def __init__(self, D, D_out):
        super().__init__()
        self.norm_final = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(D, D_out, bias=True)
        self.AdaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D, bias=True))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """DiT forward function.
        Args:
            x (Tensor): [N, T, D].
            c (Tensor): [N, D].
        Returns:
            y (Tensor): [N, T, D_out]
        """
        shift, scale = self.AdaLN_modulation.forward(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear.forward(x)
        return x


def encode_scalar(scalar: Tensor, D: int, c: float = 0.1) -> Tensor:
    """Encode scalar into sinusoidal vector.
    Args:
        scalar (Tensor): [B].
        c (float): a constant controlling the embeddings.
    Returns:
        code (Tensor): [B, D].
    """
    D_half = D // 2
    frequencies = torch.exp(-math.log(c) * torch.arange(start=0, end=D_half, dtype=torch.float32) / D_half).to(
        device=scalar.device
    )
    args = scalar[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ScalarEmbedder(nn.Module):
    def __init__(self, F: int, D: int):
        """Encode a scalar into a vector.
        Args:
            F (int): The frequency embedding size.
            D (int): The embedding size.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(F, D, bias=True), nn.SiLU(), nn.Linear(D, D, bias=True))
        self.F = F

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t (Tensor): [N].
        Returns:
            p (Tensor): [N, D].
        """
        t_code = encode_scalar(t, self.F)
        t_emb = self.mlp.forward(t_code)
        return t_emb
