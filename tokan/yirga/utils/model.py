# Copyright (c) 2025 TokAN Project
# TokAN: Token-based Accent Conversion
#
# Licensed under the MIT License - see LICENSE file for details

import torch
import torch.nn.functional as F
from einops import repeat


def generate_even_path(upsample_rate, mask):
    device = mask.device
    b, t_x, t_y = mask.shape
    path = repeat(torch.eye(t_x, device=device), "x1 x2 -> b x1 x2", b=b)  # (b, t_x, t_x)
    path = F.interpolate(path, scale_factor=upsample_rate, mode="nearest")  # (b, t_x, t_y)
    path = F.pad(path, (0, t_y - path.size(-1)))
    path = path * mask
    return path


def scale_to_total_duration(d, total_duration):
    """
    Args:
        d (torch.tensor): batch of duration values.
            shape: (B, 1, T)
        total_duration (torch.tensor): total duration values for each sample.
            shape: (B)
    Returns:
        rounded_d (torch.tensor): scaled duration values that sum up to the given total duration.
            shape: (1)
    """
    # Sum the durations along the time dimension
    sum_d = torch.sum(d, dim=-1, keepdim=True)  # shape: (B, 1, 1)

    # Avoid division by zero by replacing zeros with ones
    sum_d[sum_d == 0] = 1.0

    # Scale the durations to match the total duration
    scaled_d = d * (total_duration.view(-1, 1, 1) / sum_d)  # shape: (B, 1, T)

    # Round the scaled durations to the nearest integers
    rounded_d = torch.round(scaled_d)

    # Calculate the difference between the total duration and the sum of the rounded durations
    diff = total_duration.view(-1, 1) - torch.sum(rounded_d, dim=-1)
    diff = diff.long()

    # Adjust the rounded durations to ensure the sum matches the total duration
    for i in range(diff.size(0)):
        if diff[i] > 0:
            # If the sum of rounded durations is less than the total duration, add 1 to some elements
            indices = torch.argsort(scaled_d[i, 0] - rounded_d[i, 0], descending=True)
            rounded_d[i, 0, indices[: diff[i]]] += 1
        elif diff[i] < 0:
            # If the sum of rounded durations is more than the total duration, subtract 1 from some elements
            indices = torch.argsort(rounded_d[i, 0] - scaled_d[i, 0], descending=True)
            rounded_d[i, 0, indices[: abs(diff[i])]] -= 1

    return rounded_d
