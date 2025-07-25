# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor
from einops import repeat

from tokan.utils import get_pylogger
from tokan.matcha.models.components.text_encoder import LayerNorm
from tokan.yirga.models.components.dit_rope.model import DiT

log = get_pylogger(__name__)

DURATION_PREDICTOR_TYPES = ["regression", "flow_matching"]


def get_duration_predictor(dp_params):
    dp_type = dp_params["dp_type"]
    assert dp_type in DURATION_PREDICTOR_TYPES
    if dp_type == "regression":
        dp_cls = RegressionDurationPredictor
    if dp_type == "flow_matching":
        dp_cls = FlowMatchingDurationPredictor

    return dp_cls(**dp_params)


class ConvPredictor(nn.Module):
    def __init__(self, input_channels, n_channels, kernel_size, n_layers, p_dropout, output_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        conv_dict = OrderedDict()
        for i in range(self.n_layers):
            conv_dict.update(
                [
                    (
                        "conv1d_{}".format(i + 1),
                        nn.Conv1d(
                            self.input_channels if i == 0 else self.n_channels,
                            self.n_channels,
                            kernel_size=self.kernel_size,
                            padding=(self.kernel_size - 1) // 2,
                        ),
                    ),
                    ("relu_{}".format(i + 1), nn.ReLU()),
                    ("layer_norm_{}".format(i + 1), LayerNorm(self.n_channels)),
                    ("dropout_{}".format(i + 1), nn.Dropout(self.p_dropout)),
                ]
            )
        self.conv_layers = nn.Sequential(conv_dict)

        self.output_proj = nn.Linear(n_channels, output_channels)

    def forward(self, x, x_mask):
        """
        Args:
            x (torch.tensor): batch of tokens representations.
                shape: (B, D, T)
            x_mask (torch.tensor): batch of x's masks
                shape: (B, 1, T)
        Returns:
            x (torch.tensor): batch of representations with desired dimension
                shape: (B, ChnOut, T)
        """
        x = self.conv_layers(x) * x_mask
        x = x.transpose(1, 2).contiguous()
        o = self.output_proj(x)
        o = o.transpose(1, 2).contiguous()
        return o


class VariancePredictor(nn.Module):
    def __init__(self, input_channels, n_channels, kernel_size, n_layers, p_dropout, log_scale):
        super().__init__()
        self.log_scale = log_scale

        self.predictor = ConvPredictor(input_channels, n_channels, kernel_size, n_layers, p_dropout, output_channels=1)

        self.embedder = nn.Conv1d(
            in_channels=1,
            out_channels=input_channels,
            kernel_size=1,
            padding=0,
        )

    def compute_loss(self, x, x_mask, lin_v):
        """
        Args:
            x (torch.tensor): batch of tokens representations.
                shape: (B, D, T)
            x_mask (torch.tensor): batch of x's masks
                shape: (B, 1, T)
            lin_v (torch.tensor): batch of ground-truth linear values.
                shape: (B, T)
        """
        v_hat = self.predictor(x, x_mask).squeeze(1)  # (B, T)
        v = torch.log(lin_v + 1) if self.log_scale else lin_v

        embed = self.embed_variance(v, x_mask)  # (B, D, T)

        loss = F.mse_loss(v_hat, v, reduction="sum") / torch.sum(x_mask)

        return loss, embed

    def forward(self, x, x_mask, lin_v=None):
        """
        Args:
            x (torch.tensor): batch of tokens representations.
                shape: (B, D, T)
            x_mask (torch.tensor): batch of x's masks
                shape: (B, 1, T)
            lin_v (torch.tensor, optional): batch of ground-truth linear values.
                shape: (B, T)
        Returns:
            lin_v (torch.tensor): batch of linear values.
                shape: (B, T)
            embed (torch.tensor): batch of embeddings.
                shape: (B, D, T)
        """
        if lin_v is not None:
            v = torch.log(lin_v + 1) if self.log_scale else lin_v
        else:
            v = self.predictor(x, x_mask).squeeze(1)  # (B, T)
            lin_v = torch.exp(v) - 1

        embed = self.embed_variance(v, x_mask)  # (B, D, T)

        return lin_v, embed

    def embed_variance(self, v, mask):
        """
        Args:
            v (torch.tensor): batch of linear values.
                shape: (B, T)
            mask (torch.tensor): batch of masks.
                shape: (B, 1, T)
        Returns:
            embed (torch.tensor): batch of embeddings.
                shape: (B, D, T)
        """
        v = v.unsqueeze(1)
        embed = self.embedder(v)
        embed = embed * mask
        return embed


class RegressionDurationPredictor(nn.Module):
    def __init__(self, dp_type, input_channels, n_channels, kernel_size, n_layers, p_dropout, log_scale):
        super().__init__()
        self.dp_type = dp_type
        self.log_scale = log_scale

        self.predictor = ConvPredictor(input_channels, n_channels, kernel_size, n_layers, p_dropout, output_channels=1)

    def compute_loss(self, x, x_mask, lin_d):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, D, T)
            mask (torch.tensor): batch of masks.
                shape: (B, 1, T)
            lin_d (torch.tensor): batch of ground-truth linear duration values.
                shape: (B, 1, T)
        Returns:
            loss (torch.tensor): loss value.
                shape: (1)
        """
        d_hat = self.predictor(x, x_mask) * x_mask
        d = torch.log(lin_d + 1e-8) if self.log_scale else lin_d
        loss = F.mse_loss(d_hat, d * x_mask, reduction="sum") / torch.sum(x_mask)
        return loss

    def forward(self, x, x_mask):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, D, T)
            mask (torch.tensor): batch of masks.
                shape: (B, 1, T)
        Returns:
            lin_d (torch.tensor): predicted linear duration values
        """
        d = self.predictor(x, x_mask)
        lin_d = (torch.exp(d) - 1e-8) if self.log_scale else d
        lin_d = torch.clamp(lin_d, min=1.0) * x_mask
        return lin_d

    def round_duration(self, lin_d):
        return torch.ceil(lin_d)

    @property
    def support_total_duration(self):
        return False


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class DiTPredictor(nn.Module):
    def __init__(self, N_layer, N_head, D_hidden, D_cond, P_dropout):
        super().__init__()
        self.input_linear = nn.Linear(1 + D_cond, D_hidden)
        self.dit = DiT(D_hidden, D_hidden, N_head, N_layer, P_dropout)
        self.output_linear = nn.Linear(D_hidden, 1)

    def forward(self, x: Tensor, w: Tensor, t: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, T, 1].
            w (Tensor): [N, T, D_cond]
            t (Tensor): [N], current time.
            mask (Tensor): [N, T], attention mask, True for valid positions.
        Returns:
            y (Tensor): [N, T, D_mel], estimated mel gradient.
        """
        N, T, _ = x.shape
        p = torch.arange(0, T, 1, dtype=torch.float32, device=x.device)  # [T]
        _p = p.view(1, -1)  # [1, T]
        mask_3d = mask.unsqueeze(1).expand(-1, T, -1)  # [N, T, T]

        x = torch.cat([x, w], dim=-1)  # [N, T, 1 + D_cond]
        x = self.input_linear.forward(x)  # [N, T, D]
        x = self.dit.forward(x, _p, t, mask=mask_3d)  # [N, T, D]

        x = self.output_linear(x)

        return x


class FlowMatchingDurationPredictor(torch.nn.Module):
    def __init__(
        self,
        dp_type,
        n_layers,
        n_heads,
        n_channels,
        input_channels,
        p_dropout,
        log_scale=False,
        training_cfg_rate=0.2,
        inference_cfg_rate=0.5,
    ):
        super().__init__()

        self.average_duration_embedder = Conv(
            in_channels=1,
            out_channels=input_channels,
            kernel_size=1,
            padding=0,
        )

        self.predictor = DiTPredictor(
            N_layer=n_layers,
            N_head=n_heads,
            D_hidden=n_channels,
            D_cond=input_channels,
            P_dropout=p_dropout,
        )

        self.dp_type = dp_type
        self.log_scale = log_scale
        self.training_cfg_rate = training_cfg_rate
        self.inference_cfg_rate = inference_cfg_rate

    def forward(
        self,
        x,
        x_mask,
        total_duration=None,
        n_steps=32,
        cfg_rate=None,
        t_scheduler="cosine",
        sigma=0.0,
    ):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, D, T)
            x_mask (torch.tensor): batch of masks.
                shape: (B, 1, T)
            total_duration (torch.tensor): total duration values for each sample.
                shape: (B)
        Returns:
            lin_d (torch.tensor): predicted linear duration values
                shape: (B, 1, T)
        """
        N, D, T = x.shape

        if cfg_rate is None:
            cfg_rate = self.inference_cfg_rate

        x = x.transpose(1, 2)  # (B, T, C)
        x_lengths = x_mask.squeeze(1).sum(dim=1)  # (B)
        x_mask = x_mask.squeeze(1).bool()  # (B, T), true for non-paded positions

        if total_duration is not None:
            avg_d = total_duration / x_lengths  # (B)
            if self.log_scale:
                avg_d = torch.log(avg_d + 1e-8)
            avg_d = repeat(avg_d, "b -> b t c", t=T, c=1)  # (B, T, 1)
            avg_d_emb = self.average_duration_embedder(avg_d)  # (B, T, C)
        else:
            avg_d_emb = torch.zeros((N, T, D), device=x.device)

        cfg_x = x
        x = x + avg_d_emb

        # Create noises and time steps for the ODE/SDE process
        d = torch.randn((N, T, 1), device=self.device)
        times = torch.linspace(0, 1, n_steps + 1, device=self.device)
        if t_scheduler == "cosine":
            times = 1 - torch.cos(times * 0.5 * torch.pi)

        for idx in range(len(times) - 1):
            t0 = times[idx]
            t1 = times[idx + 1]
            dt = t1 - t0
            _t0 = torch.as_tensor([t0]).to(self.device)

            dphi_dt = self.predictor(d, x, _t0, x_mask)
            if cfg_rate > 0.0:
                cfg_dphi_dt = self.predictor(
                    d,
                    cfg_x,
                    _t0,
                    x_mask,
                )
                dphi_dt = (1.0 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt

            # Update duration values using the gradient estimate
            d = d + dphi_dt * dt

            # Add noise for SDE sampling, except at the last step
            if sigma > 0 and t1 != 1.0:
                dw = torch.randn_like(d) * math.sqrt(dt) * sigma
                d = d + dw

        lin_d = d.transpose(1, 2)  # (B, 1, T)
        if self.log_scale:
            lin_d = torch.exp(lin_d)

        lin_d = torch.clamp(lin_d, min=1.0) * x_mask.unsqueeze(1)

        return lin_d

    def compute_loss(self, x, x_mask, lin_d):
        """
        Args:
            x (torch.tensor): batch of token representations.
                shape: (B, D, T)
            x_mask (torch.tensor): batch of masks, `true` for non-padded positions.
                shape: (B, 1, T)
            lin_d (torch.tensor): batch of ground-truth linear duration values.
                shape: (B, 1, T)
        Returns:
            loss (torch.tensor): loss value.
                shape: (1)
        """
        B, D, T = x.shape

        x = x.transpose(1, 2)  # (B, T, C)
        x_lengths = x_mask.squeeze(1).sum(dim=1)  # (B)
        x_mask = x_mask.squeeze(1).bool()  # (B, T), true for non-paded positions

        d = lin_d.transpose(1, 2)  # (B, T, 1)
        if self.log_scale:
            d = torch.log(d + 1e-8)

        avg_d = lin_d.float().squeeze(1).sum(dim=1) / x_lengths  # (B)
        if self.log_scale:
            avg_d = torch.log(avg_d + 1e-8)

        avg_d = repeat(avg_d, "b -> b t c", t=T, c=1)  # (B, T, 1)
        avg_d_emb = self.average_duration_embedder(avg_d)  # (B, T, C)

        if self.training_cfg_rate > 0.0:
            cfg_mask = torch.rand(B, device=x.device) > self.training_cfg_rate
            avg_d_emb = avg_d_emb * cfg_mask.view(-1, 1, 1)  # (B, T, D)

        x = x + avg_d_emb

        z = self.get_noises((B, T, 1), x.device)  # [N, T, 1]
        t = self.get_random_times(B, device=x.device)  # [N]
        _t = t.view(B, 1, 1)
        d_t = z * (1.0 - _t) + d * _t  # [N, T, 1]

        u = d - z  # (B, T, 1)
        dphi_dt = self.predictor(d_t, x, t, x_mask)  # [N, T, 1]
        loss = F.mse_loss(
            dphi_dt.masked_select(x_mask.unsqueeze(2)), u.masked_select(x_mask.unsqueeze(2)), reduction="sum"
        ) / torch.sum(x_mask)

        return loss

    def get_random_times(self, N, device):
        t = torch.rand(N, dtype=torch.float32, device=device) * 0.999  # [N]
        return t

    def get_noises(self, shape, device):
        N, T, D = shape
        noise = torch.randn((N, T, D), device=device)
        return noise

    def round_duration(self, lin_d):
        # NOTE: the flow-matching model is trained with rounded duration targets,
        # so the generated values are naturally around integers.
        return torch.round(lin_d)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def support_total_duration(self):
        return True
