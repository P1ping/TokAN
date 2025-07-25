# Copyright (c) 2025 TokAN Project
# Original implementation by Zhijun Liu
# Adapted for TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import torch
from torch import nn, Tensor, BoolTensor
from torch.nn.functional import dropout
from typing import Tuple
from math import sqrt


def compute_r(p: Tensor, C: int, theta: float = 10000.0) -> Tensor:
    """Compute complex rotation from integer positions.
    Args:
        p (Tensor): [N, T], tensor denoting position in each sequence.
        C (int): the dimension to rotate.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        r (Tensor): [N, T, C // 2].
    """
    s = torch.arange(0, C, 2, dtype=torch.float32, device=p.device)[: (C // 2)] / C  # [C // 2]
    m = 1.0 / (theta**s)  # [C // 2]
    diff_dim = len(p.shape) - 1
    r = m.view(*(1,) * diff_dim, -1) * p.float().unsqueeze(-1)  # [N, T, C // 2]
    return r


def rotate(x: Tensor, r: Tensor) -> Tensor:
    """For different heads, the same rotation is applied.
    Args:
        x (Tensor): [N, H, T, C], where C is even and to be rotated in pairs.
        r (Tensor): [N, T, C // 2].
    Returns:
        x (Tensor): [N, H, T, C], the rotated tensor.
    """
    C = x.shape[-1]

    x_reshape = x.float().reshape(*x.shape[:-1], C // 2, 2)
    # [N, H, T, C // 2, 2]

    r = r.unsqueeze(-3)

    return (
        torch.stack(
            [
                x_reshape[..., 0] * r.cos() - x_reshape[..., 1] * r.sin(),
                x_reshape[..., 0] * r.sin() + x_reshape[..., 1] * r.cos(),
            ],
            dim=-1,
        )
        .flatten(-2, -1)
        .type_as(x)
    )


class RoPESelfAttention(nn.Module):
    def __init__(
        self,
        D: int,
        N_head: int,
        bias: bool = False,
        P_dropout: float = 0,
    ) -> None:
        super().__init__()
        assert D % N_head == 0
        self.H = N_head
        self.D = D
        self.C = D // N_head
        self.P_dropout = P_dropout

        self.linear_qkv = nn.Linear(D, 3 * D, bias=bias)
        self.linear_out = nn.Linear(D, D, bias=bias)
        self.last_drop = nn.Dropout(P_dropout)

    def compute_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute key, queue, and value from input tensor x.
        Args:
            x (Tensor): [N, T, D].
        Returns:
            q, k, v (Tensor): [N, H, T, C], C = D // H.
        """
        H, D, C = self.H, self.D, self.C
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.linear_qkv(x).split(D, dim=-1)  # [N, T, D] x 3
        try:
            q = torch.unflatten(q, -1, (H, C)).transpose(-2, -3)
            k = torch.unflatten(k, -1, (H, C)).transpose(-2, -3)
            v = torch.unflatten(v, -1, (H, C)).transpose(-2, -3)
        except AttributeError:
            N, T, _ = x.shape
            q = q.reshape(N, T, H, C).transpose(-2, -3)
            k = k.reshape(N, T, H, C).transpose(-2, -3)
            v = v.reshape(N, T, H, C).transpose(-2, -3)
        # [N, H, T, C]
        return q, k, v

    def forward(self, x: Tensor, r: Tensor, mask: BoolTensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute self-attention.
        Args:
            x (Tensor): [N, T, D].
            r (Tensor): [N, T, C // 2], where C = D // H.
            mask (BoolTensor): [N, T, T], see document on SDPA in PyTorch.
        Returns:
            y (Tensor): [N, T, D].
            k, v (Tensor): [N, H, T, C].
        """
        mask_h = mask.unsqueeze(-3).expand(-1, self.H, -1, -1)  # [N, H, T, T]
        q, k, v = self.compute_qkv(x)  # [N, H, T, C]

        q = rotate(q, r)
        k = rotate(k, r)

        try:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                o = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.P_dropout if self.training else 0,
                    attn_mask=mask_h,
                )  # [N, H, T, C]
        except:
            w = q @ k.transpose(-1, -2) / sqrt(self.C)  # [N, ..., H, T, T]
            w = w.masked_fill(~mask_h, float("-inf"))
            attn = torch.softmax(w, dim=-1)
            if self.training:
                attn = dropout(attn, p=self.P_dropout)
            o = attn @ v

        y = o.transpose(-2, -3).contiguous().flatten(-2, -1)
        # [N, T, D]
        y = self.linear_out.forward(y)
        y = self.last_drop.forward(y)
        return y, k, v

    def get_attn(self, x: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """Compute the attention weights.
        Args: Same as forward(...)
        Returns:
            attn (Tensor): [N, H, T, T].
        """
        with torch.no_grad():
            q, k, _ = self.compute_qkv(x)  # [N, ..., H, T, C]
            q = rotate(q, r)
            k = rotate(k, r)
            w = q @ k.transpose(-1, -2) / sqrt(self.C)  # [N, ..., H, T, T]
            w = w.masked_fill(~mask, float("-inf"))
            return torch.softmax(w, dim=-1)
