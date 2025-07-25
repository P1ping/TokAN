# Copyright (c) 2025 TokAN Project
# Original implementation by Zhijun Liu
# Adapted for TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import torch
from torch import nn, Tensor, BoolTensor
from torch.nn import Conv2d

from .model import DiT


LRELU_SLOPE = 0.1


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        """
        Args:
            channels (int): The number of input channels.
            out_channels (int, optional): The number of output channels. Defaults to `channels`.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            Conv2d(channels, channels, kernel, padding=kernel // 2),
            nn.LeakyReLU(LRELU_SLOPE),
            Conv2d(channels, channels, kernel, padding=kernel // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """[BCHW] -> [BCHW]"""
        h = self.layers(x)
        return x + h


class DiTDecoder(nn.Module):
    def __init__(
        self,
        N_layer: int,
        N_head: int,
        D: int,
        D_cond: int,
        D_hidden: int,
        D_bridge: int,
        D_conv2d: int,
        D_mel: int,
        P_dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()
        self.D_mel = D_mel
        self.D_bridge = D_bridge

        self.input_linear = nn.Linear(D_mel + D_cond, D)
        self.dit = DiT(D, D_hidden, N_head, N_layer, P_dropout)

        if D_bridge > 0:
            assert D_conv2d > 0, '"D_conv2d" should be greater than 0 when "D_bridge" set.'
            self.bridge_linear = nn.Linear(D, D_bridge * D_mel)
            self.cond_linear = nn.Linear(D_cond, D_mel)
            self.bridge_conv2d = Conv2d(D_bridge + 2, D_conv2d, 3, 1, 1)
            self.resblock_2d = nn.Sequential(ResBlock2d(D_conv2d, kernel=3), ResBlock2d(D_conv2d, kernel=3))
            self.last_conv2d = nn.Sequential(nn.LeakyReLU(LRELU_SLOPE), Conv2d(D_conv2d, 1, 3, 1, 1))
            self.output_linear = None
        else:
            self.bridge_linear = None
            self.bridge_conv2d = None
            self.cond_linear = None
            self.resblock_2d = None
            self.last_conv2d = None
            self.output_linear = nn.Linear(D, D_mel)

    def forward(self, x: Tensor, w: Tensor, t: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, T, D_mel].
            w (Tensor): [N, T, D_cond]
            t (Tensor): [N], current time.
            mask (Tensor): [N, T], attention mask, True for valid positions.
        Returns:
            y (Tensor): [N, T, D_mel], estimated mel gradient.
        """
        N, T, D_mel = x.shape
        p = torch.arange(0, T, 1, dtype=torch.float32, device=x.device)  # [T]
        _p = p.view(1, -1)  # [1, T]
        mask_3d = mask.unsqueeze(1).expand(-1, T, -1)  # [N, T, T]

        x_res = x
        x = torch.cat([x, w], dim=-1)  # [N, T, D_mel + D_cond]
        x = self.input_linear.forward(x)  # [N, T, D]
        x = self.dit.forward(x, _p, t, mask=mask_3d)  # [N, T, D]

        # Direct output without residual post-processing
        if self.D_bridge <= 0:
            x = self.output_linear(x)
            return x

        # Post-processing using residual block
        x = self.bridge_linear.forward(x)  # [N, T, D_bridge * D_mel]
        x = torch.unflatten(x, dim=-1, sizes=(self.D_bridge, D_mel))
        # [N, T, D_bridge, D_mel]
        x = x.transpose(-2, -3).contiguous()  # [N, D_bridge, T, D_mel]
        w_mel = self.cond_linear(w)  # [N, T, D_mel]
        x = torch.cat([x_res.unsqueeze(-3), w_mel.unsqueeze(-3), x], dim=-3)  # [N, D_bridge + 2, T, D_mel]

        x = self.bridge_conv2d.forward(x)
        x = self.resblock_2d.forward(x)
        x = self.last_conv2d.forward(x)  # [N, 1, T, D_mel]

        return x.squeeze(-3)

    def get_attn(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        """Inputs are the same as in forward(...).
        Returns:
            attn (Tensor): [N, Layers, H, T, T].
        """
        N, T, _ = x.shape
        p = torch.arange(0, T, 1, dtype=torch.float32, device=x.device)
        _p = p.view(1, -1)
        mask_3d = mask.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([x, c], dim=-1)  # [N, T, 2 x D_mel]
        x = self.input_linear.forward(x)  # [N, T, D]
        attn = self.dit.get_attn(x, _p, t, mask=mask_3d)  # [N, T, D]

        return attn
