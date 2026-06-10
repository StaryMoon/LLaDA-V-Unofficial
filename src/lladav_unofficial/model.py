"""Compact unofficial PyTorch implementation for LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning.

This module is intentionally small. It is not a faithful reproduction yet; it
provides named components, tensor interfaces, and replacement points for a complete reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StarterConfig:
    image_channels: int = 3
    vocab_size: int = 4096
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    output_dim: int = 3


class PatchOrStepEncoder(nn.Module):
    """Encode images, videos, or symbolic steps into compact tokens."""

    def __init__(self, config: StarterConfig) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(config.image_channels, config.hidden_dim, kernel_size=4, stride=4)
        self.step_proj = nn.Linear(config.image_channels, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            tokens = self.image_proj(x).flatten(2).transpose(1, 2)
            tokens = tokens.mean(dim=1).reshape(b, t, -1)
            return tokens
        if x.dim() == 4:
            return self.image_proj(x).flatten(2).transpose(1, 2)
        if x.dim() == 3:
            return self.step_proj(x)
        raise ValueError(f"expected 3D, 4D, or 5D tensor, got {tuple(x.shape)}")


class MultimodalConditioner(nn.Module):
    """Fuse visual or temporal tokens with optional text/instruction tokens."""

    def __init__(self, config: StarterConfig) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)

    def forward(self, tokens: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_ids is not None:
            text_tokens = self.text_embedding(token_ids)
            tokens = torch.cat([text_tokens, tokens], dim=1)
        return self.encoder(tokens)


class MaskedDiffusionHead(nn.Module):
    """Task head for diffusion-style multimodal visual instruction tuning.

    The output interface is compact by default and is designed to be replaced with the full paper-specific target representation.
    """

    def __init__(self, config: StarterConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.summary = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.norm(tokens).mean(dim=1)
        return self.out(F.gelu(self.summary(pooled)))


class UnofficialModel(nn.Module):
    """PyTorch model interface for LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning."""

    def __init__(self, config: Optional[StarterConfig] = None) -> None:
        super().__init__()
        self.config = config or StarterConfig()
        self.encoder = PatchOrStepEncoder(self.config)
        self.conditioner = MultimodalConditioner(self.config)
        self.head = MaskedDiffusionHead(self.config)

    def forward(self, x: torch.Tensor, token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.encoder(x)
        fused = self.conditioner(tokens, token_ids=token_ids)
        return self.head(fused)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Default regression loss used by the minimal training example."""
    return F.smooth_l1_loss(prediction, target)
