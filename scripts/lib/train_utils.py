"""Вспомогательные функции обучения WGAN."""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def setup_logging(logs_dir: Path, experiment_name: str) -> logging.Logger:
    """Настраивает логгер с выводом в файл и консоль."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{experiment_name}.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("wgan_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Считает gradient penalty для WGAN-GP."""
    batch = real.size(0)
    eps = torch.rand(batch, 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    d_interp = discriminator(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def compute_metrics(w_pred: np.ndarray, w_true: np.ndarray) -> Tuple[float, float, float]:
    """Вычисляет RMSE, MAE и R2 между предсказанием и истиной."""
    a = w_pred.reshape(-1)
    b = w_true.reshape(-1)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    ss_res = np.sum((b - a) ** 2)
    ss_tot = np.sum((b - np.mean(b)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return rmse, mae, r2
