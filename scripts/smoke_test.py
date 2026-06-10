#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from lladav_unofficial import StarterConfig, UnofficialModel, reconstruction_loss


def main() -> None:
    config = StarterConfig(hidden_dim=64, num_layers=1, num_heads=4, output_dim=3)
    model = UnofficialModel(config)
    x = torch.randn(2, 3, 64, 64)
    token_ids = torch.randint(0, config.vocab_size, (2, 8))
    out = model(x, token_ids=token_ids)
    target = torch.zeros_like(out)
    loss = reconstruction_loss(out, target)
    print("output:", tuple(out.shape))
    print("loss:", float(loss.detach()))


if __name__ == "__main__":
    main()
