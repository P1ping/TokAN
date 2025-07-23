from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from einops import repeat, pack


class CombinedTimestepCondEmbeddings(nn.Module):
    def __init__(self, embedding_dim, time_dim=0, cond_dim=0, cond_dropout=0.0):
        super().__init__()
        input_dim = time_dim + cond_dim
        assert input_dim > 0
        self.embedder = nn.Linear(input_dim, embedding_dim)
        self.cond_dropout = cond_dropout

    def forward(self, timestep: torch.Tensor = None, cond: torch.Tensor = None) -> torch.Tensor:
        if cond is not None and self.cond_dropout > 0 and self.training:
            dropout_mask = torch.empty((cond.size(0), 1), device=cond.device).fill_(1 - self.cond_dropout)
            dropout_mask = repeat(torch.bernoulli(dropout_mask), "n 1 -> n d", d=cond.size(1))
            cond = cond * dropout_mask

        if (timestep is not None) and (cond is not None):
            conditioning, _ = pack([timestep, cond], "b *")
        elif timestep is not None:
            conditioning = timestep
        elif cond is not None:
            conditioning = cond
        else:
            raise ValueError("At least one of timestep or cond must be provided")

        conditioning = self.embedder(conditioning)

        return conditioning


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        cond_dim (`int`): The size of each conditioning vector.
    """

    def __init__(self, embedding_dim: int, time_dim: int, cond_dim: int, cond_dropout=0.1):
        super().__init__()
        self.emb = CombinedTimestepCondEmbeddings(embedding_dim, time_dim, cond_dim, cond_dropout)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep, cond)))
        scale, shift = emb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        cond_dim (`int`): The size of each conditioning vector.
    """

    def __init__(self, embedding_dim: int, time_dim: int, cond_dim: int, cond_dropout=0.1):
        super().__init__()

        self.emb = CombinedTimestepCondEmbeddings(embedding_dim, time_dim, cond_dim, cond_dropout)

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, cond)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


if __name__ == "__main__":
    # Mock inputs
    batch_size = 5  # Example batch size
    length = 10  # Example sequence length
    time_dim = 128  # Example time dimension
    cond_dim = 10  # Example conditioning dimension
    embedding_dim = 20  # Example embedding dimension

    x = torch.rand(batch_size, length, embedding_dim)  # Mock input tensor
    timesteps = torch.rand((batch_size, time_dim))
    cond = torch.randn(batch_size, cond_dim)  # Random conditioning vectors

    # Instantiate the CombinedTimestepCondEmbeddings class
    model = AdaLayerNormZero(embedding_dim=embedding_dim, time_dim=time_dim, cond_dim=cond_dim)

    # Ensure model is in evaluation mode to disable dropout for consistent output
    model.train()

    # Pass the mock data through the model
    output = model(x, timesteps, cond)

    # Print the output shape to verify it works as expected
    for item in output:
        print(item.shape)
