#!/usr/bin/env python3
"""
Скрипт обучения WGAN для калибровки калориметра.

Пайплайн:
1. Загрузка конфига
2. Получение датасета с aging factor
3. Фильтрация датасета по xyz
4. Обучение WGAN
5. Сохранение лучшей модели и построение графиков
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from comet_ml import Experiment

# Добавляем корень проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from utils import AgingFactorGenerator, filter_by_xyz
from plot_graphics import plot_true_vs_predicted, plot_error_histogram

import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt


# ==============================================================================
# Логирование
# ==============================================================================

def setup_logging(logs_dir: Path, experiment_name: str) -> logging.Logger:
    """Настройка логирования в файл и консоль."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"{experiment_name}.log"
    
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем логгер
    logger = logging.getLogger('wgan_training')
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие хендлеры
    logger.handlers.clear()
    
    # Хендлер для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Хендлер для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ==============================================================================
# Dataset
# ==============================================================================

class CaloEventDataset(Dataset):
    """Dataset для событий калориметра."""
    
    def __init__(self, df: pd.DataFrame, x_map: dict, y_map: dict, z_map: dict, 
                 nx: int, ny: int, nz: int, event_col: str = 'event'):
        self.events = []
        self.x_map = x_map
        self.y_map = y_map
        self.z_map = z_map
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        for eid, ev in df.groupby(event_col, sort=False):
            self.events.append(ev[['x', 'y', 'z', 'E_new', 'E_old']].to_numpy())
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        ev = self.events[idx]
        
        E_new = torch.zeros(self.nz, self.nx, self.ny)
        E_old = torch.zeros(self.nz, self.nx, self.ny)
        mask = torch.zeros(self.nz, self.nx, self.ny)
        
        for x, y, z, en, eo in ev:
            iz = self.z_map.get(z)
            ix = self.x_map.get(x)
            iy = self.y_map.get(y)
            
            if iz is not None and ix is not None and iy is not None:
                E_new[iz, ix, iy] = float(en)
                E_old[iz, ix, iy] = float(eo)
                mask[iz, ix, iy] = 1.0
        
        return {'E_new': E_new, 'E_old': E_old, 'mask': mask}


# ==============================================================================
# Models
# ==============================================================================

class Generator(nn.Module):
    """Генератор - обучаемый тензор aging factor."""
    
    def __init__(self, shape: Tuple[int, int, int], eps: float = 1e-8):
        super().__init__()
        self.W = nn.Parameter(torch.ones(shape))
        self.eps = eps
    
    def forward(self, E_old: torch.Tensor) -> torch.Tensor:
        B = E_old.size(0)
        W = self.W.unsqueeze(0).expand(B, -1, -1, -1)
        return E_old / (W + self.eps)


class Discriminator(nn.Module):
    """Дискриминатор для WGAN."""
    
    def __init__(self, in_dim: int, dim: int = 64):
        super().__init__()
        
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.InstanceNorm2d(o, affine=True),
                nn.LeakyReLU(0.2)
            )
        
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            block(dim, dim * 2),
            block(dim * 2, dim * 2),
            block(dim * 2, dim),
            nn.Conv2d(dim, 1, 4),
            nn.AdaptiveMaxPool2d(1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# ==============================================================================
# Training utilities
# ==============================================================================

def gradient_penalty(D: nn.Module, real: torch.Tensor, fake: torch.Tensor, 
                     device: torch.device) -> torch.Tensor:
    """Вычисление gradient penalty для WGAN-GP."""
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_(True)
    
    d_interp = D(interp)
    
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def build_real_aging_tensor(df: pd.DataFrame, x_map: dict, y_map: dict, z_map: dict,
                            nx: int, ny: int, nz: int) -> torch.Tensor:
    """Построение тензора реальных aging factor."""
    real_W = torch.ones(nz, nx, ny)
    
    cell_df = (
        df[['cell_key', 'x', 'y', 'z', 'aging_factor']]
        .drop_duplicates('cell_key')
        .reset_index(drop=True)
    )
    
    for _, row in cell_df.iterrows():
        x, y, z = row[['x', 'y', 'z']]
        a = row['aging_factor']
        
        iz = z_map.get(z)
        ix = x_map.get(x)
        iy = y_map.get(y)
        
        if iz is not None and ix is not None and iy is not None:
            real_W[iz, ix, iy] = float(a)
    
    return real_W


def compute_metrics(W_pred: np.ndarray, W_true: np.ndarray) -> Tuple[float, float]:
    """Вычисление RMSE и MAE."""
    a = W_pred.reshape(-1)
    b = W_true.reshape(-1)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mae = float(np.mean(np.abs(a - b)))
    return rmse, mae


def build_prediction_dataframe(W_pred: np.ndarray, W_true: np.ndarray,
                               x_vals: np.ndarray, y_vals: np.ndarray, 
                               z_vals: np.ndarray) -> pd.DataFrame:
    """Построение DataFrame с предсказаниями для визуализации."""
    records = []
    nz, nx, ny = W_pred.shape
    
    for iz in range(nz):
        for ix in range(nx):
            for iy in range(ny):
                a_true = W_true[iz, ix, iy]
                a_pred = W_pred[iz, ix, iy]
                
                # Пропускаем ячейки с a_true == 1 (не было данных)
                if abs(a_true - 1.0) < 1e-6:
                    continue
                
                records.append({
                    'x': x_vals[ix],
                    'y': y_vals[iy],
                    'z': z_vals[iz],
                    'a_true': a_true,
                    'a_pred': a_pred
                })
    
    return pd.DataFrame(records)


# ==============================================================================
# Plotting
# ==============================================================================

def save_training_curves(history: Dict, save_path: Path):
    """Сохранение графиков обучения."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history['D_loss']) + 1)
    
    # D loss
    axes[0, 0].plot(epochs, history['D_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Discriminator Loss')
    axes[0, 0].set_title('Discriminator Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # G loss
    axes[0, 1].plot(epochs, history['G_loss'], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Generator Loss')
    axes[0, 1].set_title('Generator Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(epochs, history['W_rmse'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Weight RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[1, 1].plot(epochs, history['W_mae'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Weight MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_true_vs_predicted(df: pd.DataFrame, true_col: str, pred_col: str,
                           save_path: Path, title: str = "True vs Predicted"):
    """Сохранение scatter plot истинных vs предсказанных значений."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df[true_col], df[pred_col], s=8, alpha=0.5)
    
    # Диагональ
    lims = [
        min(df[true_col].min(), df[pred_col].min()),
        max(df[true_col].max(), df[pred_col].max())
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Ideal')
    
    ax.set_xlabel(r"True aging factor $a_i$", fontsize=12)
    ax.set_ylabel(r"Predicted $\hat{a}_i$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_error_histogram(df: pd.DataFrame, true_col: str, pred_col: str,
                         save_path: Path, title: str = "Error Distribution"):
    """Сохранение гистограммы распределения ошибок."""
    errors = df[true_col] - df[pred_col]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(errors, bins=60, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Статистики
    mean_err = errors.mean()
    std_err = errors.std()
    
    ax.axvline(mean_err, color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_err:.4f}')
    ax.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=2,
               label=f'Std: {std_err:.4f}')
    ax.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=2)
    
    ax.set_xlabel(r"Error: $a_i - \hat{a}_i$", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train_wgan(config: dict, logger: logging.Logger, experiment_name: str):
    """Основной цикл обучения WGAN."""
    
    # Параметры из конфига
    wgan_cfg = config.get('wgan_params', {})
    aging_cfg = config.get('aging', {})
    
    use_gp = wgan_cfg.get('use_gradient_penalty', False)
    gp_lambda = wgan_cfg.get('gp_lambda', 10.0)
    clip_value = wgan_cfg.get('clip_value', 0.01)
    batch_size = wgan_cfg.get('batch_size', 64)
    n_epochs = wgan_cfg.get('epochs', 30)
    lr = wgan_cfg.get('lr', 2e-4)
    betas = tuple(wgan_cfg.get('betas', [0.5, 0.999]))
    seed = wgan_cfg.get('seed', 42)
    
    filter_cfg = wgan_cfg.get('filter', {})
    x_range = tuple(filter_cfg.get('x_range', [-30, 30]))
    y_range = tuple(filter_cfg.get('y_range', [-30, 30]))
    z_range = tuple(filter_cfg.get('z_range', [-100, -80]))
    
    experiment = Experiment(
        experiment_name=f"wgan_use_gp:{use_gp}_batch_size:{batch_size}_n_events:{aging_cfg.get("n_events")}",
        auto_param_logging=False,
        auto_metric_logging=False,
        log_env_details=False,
    )

    experiment.log_parameters({
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "gp_lambda": gp_lambda ,
        "use_gradient_penalty": use_gp,
        "clip_value": clip_value,
    })

    global_step = 0

    # Директории с поддиректорией эксперимента
    base_dir = Path(config.get('results_dir', 'results')).parent
    logs_dir = base_dir / wgan_cfg.get('logs_dir', 'logs') / experiment_name
    models_dir = base_dir / wgan_cfg.get('models_dir', 'models') / experiment_name
    results_dir = base_dir / 'results' / experiment_name
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_best = wgan_cfg.get('save_best_model', True)
    save_every = wgan_cfg.get('save_every_n_epochs', 10)
    
    # Установка seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    # ============================================
    # 1. Загрузка данных
    # ============================================
    logger.info("Загрузка данных...")
    df = pd.read_csv(config['data_path'])
    logger.info(f"Загружено {len(df)} записей")
    
    # ============================================
    # 2. Генерация aging factor
    # ============================================
    logger.info("Генерация aging factor...")
    generator = AgingFactorGenerator(
        aging_function=aging_cfg.get("function", "xyz"),
        af_min=aging_cfg.get("af_min", 0.8),
        af_max=aging_cfg.get("af_max", 1.0),
        sigma=aging_cfg.get("sigma", 0.0),
        noise_seed=aging_cfg.get("noise_seed"),
        event_col=aging_cfg.get("event_col", "event"),
        x_col=aging_cfg.get("x_col", "x"),
        y_col=aging_cfg.get("y_col", "y"),
        z_col=aging_cfg.get("z_col", "z"),
        energy_col=aging_cfg.get("energy_col", "E"),
    )
    aged = generator.generate(df)
    subset = generator.sample_events(aged, n_events=aging_cfg.get("n_events"), seed=aging_cfg.get("noise_seed"))
    dataset = subset[[generator.event_col, "cell_key", generator.x_col, generator.y_col, 
                      generator.z_col, "E_new", "E_old", "aging_factor"]]
    logger.info(f"После генерации aging factor: {len(dataset)} записей")
    
    # ============================================
    # 3. Фильтрация по xyz
    # ============================================
    logger.info(f"Фильтрация по координатам: x={x_range}, y={y_range}, z={z_range}")
    dataset = filter_by_xyz(dataset, x_range, y_range, z_range)
    logger.info(f"После фильтрации: {len(dataset)} записей")
    
    # ============================================
    # 4. Подготовка данных для обучения
    # ============================================
    x_vals = np.sort(dataset['x'].unique())
    y_vals = np.sort(dataset['y'].unique())
    z_vals = np.sort(dataset['z'].unique())
    
    NX, NY, NZ = len(x_vals), len(y_vals), len(z_vals)
    
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    z_map = {v: i for i, v in enumerate(z_vals)}
    
    logger.info(f"Размерность сетки: NX={NX}, NY={NY}, NZ={NZ}")
    logger.info(f"X range: [{x_vals[0]:.2f}, {x_vals[-1]:.2f}]")
    logger.info(f"Y range: [{y_vals[0]:.2f}, {y_vals[-1]:.2f}]")
    logger.info(f"Z range: [{z_vals[0]:.2f}, {z_vals[-1]:.2f}]")
    
    # Создание dataset и dataloader
    train_ds = CaloEventDataset(dataset, x_map, y_map, z_map, NX, NY, NZ, 
                                event_col=generator.event_col)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    logger.info(f"Количество событий: {len(train_ds)}")
    logger.info(f"Количество батчей: {len(loader)}")
    
    # Реальный тензор aging factor
    logger.info("Построение реального тензора aging factor...")
    real_W = build_real_aging_tensor(dataset, x_map, y_map, z_map, NX, NY, NZ)
    real_W_np = real_W.numpy()
    
    # ============================================
    # 5. Инициализация моделей
    # ============================================
    G = Generator((NZ, NX, NY)).to(device)
    D = Discriminator(in_dim=NZ).to(device)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    
    logger.info(f"Параметры обучения:")
    logger.info(f"  - Gradient penalty: {use_gp}")
    if use_gp:
        logger.info(f"  - GP lambda: {gp_lambda}")
    else:
        logger.info(f"  - Weight clipping: {clip_value}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Epochs: {n_epochs}")
    logger.info(f"  - Learning rate: {lr}")
    
    # ============================================
    # 6. Обучение
    # ============================================
    history = {'D_loss': [], 'G_loss': [], 'W_rmse': [], 'W_mae': []}
    best_rmse = float('inf')
    best_epoch = 0
    best_W = None
    
    logger.info("=" * 60)
    logger.info("Начало обучения WGAN")
    logger.info("=" * 60)
    
    for epoch in range(n_epochs):
        D_losses = []
        G_losses = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for batch in pbar:
            E_new = batch['E_new'].to(device)
            E_old = batch['E_old'].to(device)
            
            # Маскирование пустых ячеек
            mask = ((E_new != 0) & (E_old != 0)).float()
            E_new = E_new * mask
            E_old = E_old * mask
            
            # === Обучение дискриминатора ===
            opt_D.zero_grad()
            
            fake = G(E_old).detach()
            d_real = D(E_new)
            d_fake = D(fake)
            
            d_loss = -d_real.mean() + d_fake.mean()
            
            if use_gp:
                gp = gradient_penalty(D, E_new, fake, device)
                d_loss = d_loss + gp_lambda * gp
            
            d_loss.backward()
            opt_D.step()
            
            # Weight clipping (если не используется GP)
            if not use_gp:
                for p in D.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            
            # === Обучение генератора ===
            opt_G.zero_grad()
            fake = G(E_old)
            g_loss = -D(fake).mean()
            g_loss.backward()
            opt_G.step()
            
            # Клиппинг весов генератора в [0, 1]
            with torch.no_grad():
                G.W.clamp_(0.0, 1.0)
            
            D_losses.append(float(d_loss.item()))
            G_losses.append(float(g_loss.item()))
            pbar.set_postfix(D=np.mean(D_losses), G=np.mean(G_losses))
            
            experiment.log_metrics(
                {
                    "train/D_loss_step": float(d_loss.item()),
                    "train/G_loss_step": float(g_loss.item()),
                },
                step=global_step,
                epoch=epoch,
            )
            global_step += 1
        
        # Вычисление метрик
        with torch.no_grad():
            W_pred = G.W.detach().cpu().numpy()
        w_rmse, w_mae = compute_metrics(W_pred, real_W_np)
        
        # Сохранение истории
        history['D_loss'].append(float(np.mean(D_losses)))
        history['G_loss'].append(float(np.mean(G_losses)))
        history['W_rmse'].append(w_rmse)
        history['W_mae'].append(w_mae)
        
        experiment.log_metrics(
            {
                "train/D_loss_epoch": history["D_loss"][-1],
                "train/G_loss_epoch": history["G_loss"][-1],
                "metrics/W_rmse": w_rmse,
                "metrics/W_mae": w_mae,
            },
            epoch=epoch,
        )
        
        experiment.end()
        
        logger.info(f"Epoch {epoch+1}/{n_epochs}: D={history['D_loss'][-1]:.4f}  "
                   f"G={history['G_loss'][-1]:.4f}  W_RMSE={w_rmse:.4f}  W_MAE={w_mae:.4f}")
        
        # Сохранение лучшей модели
        if save_best and w_rmse < best_rmse:
            best_rmse = w_rmse
            best_epoch = epoch + 1
            best_W = W_pred.copy()
            
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'optimizer_D_state_dict': opt_D.state_dict(),
                'history': history,
                'W_rmse': w_rmse,
                'W_mae': w_mae,
            }, models_dir / 'best_model.pt')
            
            logger.info(f"  -> Новая лучшая модель сохранена (RMSE: {w_rmse:.4f})")
        
        # Периодическое сохранение
        if save_every and (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'optimizer_D_state_dict': opt_D.state_dict(),
                'history': history,
            }, models_dir / f'checkpoint_epoch_{epoch+1}.pt')
            logger.info(f"  -> Checkpoint сохранен: checkpoint_epoch_{epoch+1}.pt")
    
    logger.info("=" * 60)
    logger.info("Обучение завершено!")
    logger.info(f"Лучшая модель: epoch {best_epoch}, RMSE={best_rmse:.4f}")
    logger.info("=" * 60)
    
    # ============================================
    # 7. Построение графиков по лучшей модели
    # ============================================
    logger.info("Построение графиков...")
    
    # Кривые обучения
    save_training_curves(history, results_dir / 'wgan_training_curves.png')
    logger.info("  -> Сохранены кривые обучения: wgan_training_curves.png")
    
    # Построение DataFrame с предсказаниями
    if best_W is not None:
        pred_df = build_prediction_dataframe(best_W, real_W_np, x_vals, y_vals, z_vals)
        
        # True vs Predicted
        save_true_vs_predicted(
            pred_df, 'a_true', 'a_pred',
            results_dir / 'wgan_true_vs_predicted.png',
            title=f"WGAN: True vs Predicted (Best epoch {best_epoch}, RMSE={best_rmse:.4f})"
        )
        logger.info("  -> Сохранен график: wgan_true_vs_predicted.png")
        
        # Error histogram
        save_error_histogram(
            pred_df, 'a_true', 'a_pred',
            results_dir / 'wgan_error_histogram.png',
            title=f"WGAN: Error Distribution (Best epoch {best_epoch})"
        )
        logger.info("  -> Сохранен график: wgan_error_histogram.png")
        
        # Сохраняем предсказания в CSV
        pred_df.to_csv(results_dir / 'wgan_predictions.csv', index=False)
        logger.info("  -> Сохранены предсказания: wgan_predictions.csv")
    
    # Финальная модель
    torch.save({
        'epoch': n_epochs,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizer_G_state_dict': opt_G.state_dict(),
        'optimizer_D_state_dict': opt_D.state_dict(),
        'history': history,
        'best_epoch': best_epoch,
        'best_rmse': best_rmse,
    }, models_dir / 'final_model.pt')
    logger.info(f"  -> Финальная модель сохранена: final_model.pt")
    
    return history, best_rmse, best_epoch


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Обучение WGAN для калибровки калориметра')
    parser.add_argument('--config', type=str, 
                        default=str(project_root / 'configs' / 'config.yaml'),
                        help='Путь к файлу конфигурации')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Название эксперимента (для логов)')
    args = parser.parse_args()
    
    # Загрузка конфига
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Название эксперимента
    if args.experiment:
        experiment_name = args.experiment
    else:
        experiment_name = datetime.now().strftime('wgan_%Y%m%d_%H%M%S')
    
    # Настройка логирования (в поддиректорию эксперимента)
    base_dir = Path(config.get('results_dir', 'results')).parent
    logs_dir = base_dir / config.get('wgan_params', {}).get('logs_dir', 'logs') / experiment_name
    logger = setup_logging(logs_dir, experiment_name)
    
    logger.info("=" * 60)
    logger.info(f"WGAN Training - {experiment_name}")
    logger.info("=" * 60)
    logger.info(f"Конфиг: {args.config}")
    
    try:
        history, best_rmse, best_epoch = train_wgan(config, logger, experiment_name)
        logger.info("Обучение успешно завершено!")
        return 0
    except Exception as e:
        logger.exception(f"Ошибка при обучении: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
