"""Модели генератора и дискриминатора для WGAN."""

from typing import Tuple

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Генератор в виде обучаемого тензора aging-факторов."""

    def __init__(self, shape: Tuple[int, int, int], eps: float = 1e-8):
        super().__init__()
        self.w = nn.Parameter(torch.ones(shape))
        self.eps = eps

    def forward(self, e_old: torch.Tensor) -> torch.Tensor:
        """Генерирует восстановленную энергию из E_old."""
        batch = e_old.size(0)
        w = self.w.unsqueeze(0).expand(batch, -1, -1, -1)
        return e_old / (w + self.eps)


class Discriminator(nn.Module):
    """Дискриминатор WGAN на сверточных блоках."""

    def __init__(self, in_dim: int, dim: int = 64, final_pool: str = "max"):
        super().__init__()

        def block(inp, out):
            return nn.Sequential(
                nn.Conv2d(inp, out, 3, padding=1),
                nn.InstanceNorm2d(out, affine=True),
                nn.LeakyReLU(0.2),
            )

        if final_pool == "max":
            pool = nn.AdaptiveMaxPool2d(1)
        elif final_pool == "mean":
            pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(
                f"Unknown final_pool={final_pool!r}, expected 'max' or 'mean'"
            )

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            block(dim, dim * 2),
            block(dim * 2, dim * 2),
            block(dim * 2, dim),
            nn.Conv2d(dim, 1, 4),
            pool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает скалярные оценки дискриминатора."""
        return self.net(x).view(-1)
