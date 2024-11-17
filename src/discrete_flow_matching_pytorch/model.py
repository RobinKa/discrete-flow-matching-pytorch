import torch
from torch import nn


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Reshape(nn.Module):
    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class ConvNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_timesteps: int,
        num_layers: int,
        kernel_size: int = 31,
    ):
        super().__init__()

        # x: B, L

        # Input embedding, B L -> B L C
        self.input_projection = nn.Embedding(vocab_size, hidden_dim)

        # Embed timestep to B, 1, C
        self.embed_timestep = nn.Sequential(
            nn.Embedding(num_timesteps, hidden_dim),
            Unsqueeze(1),
        )

        self.blocks = nn.ModuleList()
        self.timestep_embedding_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size=kernel_size, padding="same"
                    ),
                    Transpose(),
                    nn.LayerNorm([hidden_dim]),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm([hidden_dim]),
                    nn.GELU(),
                ),
            )
            self.timestep_embedding_norms.append(nn.LayerNorm([hidden_dim]))

        # Output projection, B L C -> B L V
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: B, L, V, t: B
        x = self.input_projection(x)  # BLC

        for block, timestep_embedding_norm in zip(
            self.blocks, self.timestep_embedding_norms
        ):
            x = x + block(x + timestep_embedding_norm(self.embed_timestep(t)))  # BLC

        x = self.output_projection(x)  # BLV

        return x
