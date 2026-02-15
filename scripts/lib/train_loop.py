"""Основной цикл обучения WGAN."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from comet_ml import Experiment
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib.dataset import CaloEventDataset, build_prediction_dataframe, build_real_aging_tensor
from lib.models import Discriminator, Generator
from lib.plots import (
    save_error_histogram,
    save_training_curves,
    save_training_history_csv,
    save_true_vs_predicted,
)
from lib.train_utils import compute_metrics, gradient_penalty
from utils import AgingFactorGenerator, filter_by_xyz


def train_wgan(config: dict, logger: logging.Logger, experiment_name: str, run_name: str):
    """Запускает полный цикл обучения и сохранения артефактов."""
    wgan_cfg = config.get("wgan_params", {})
    aging_cfg = config.get("aging", {})

    use_gp = wgan_cfg.get("use_gradient_penalty", False)
    gp_lambda = wgan_cfg.get("gp_lambda", 10.0)
    clip_value = wgan_cfg.get("clip_value", 0.01)
    batch_size = wgan_cfg.get("batch_size", 64)
    n_epochs = wgan_cfg.get("epochs", 30)
    lr = wgan_cfg.get("lr", 2e-4)
    betas = tuple(wgan_cfg.get("betas", [0.5, 0.999]))
    seed = wgan_cfg.get("seed", 42)
    log_every_n_steps = int(wgan_cfg.get("log_every_n_steps", 100))

    filter_cfg = wgan_cfg.get("filter", {})
    x_range = tuple(filter_cfg.get("x_range", [-30, 30]))
    y_range = tuple(filter_cfg.get("y_range", [-30, 30]))
    z_range = tuple(filter_cfg.get("z_range", [-100, -80]))

    experiment = Experiment(
        experiment_name=experiment_name,
        auto_param_logging=False,
        auto_metric_logging=False,
        log_env_details=False,
    )
    experiment.log_parameters(
        {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "gp_lambda": gp_lambda,
            "use_gradient_penalty": use_gp,
            "clip_value": clip_value,
        }
    )

    global_step = 0
    base_dir = Path(config.get("results_dir", "results")).parent
    logs_dir = base_dir / wgan_cfg.get("logs_dir", "logs") / run_name / experiment_name
    models_dir = base_dir / wgan_cfg.get("models_dir", "models") / run_name / experiment_name
    results_root_name = Path(config.get("results_dir", "results")).name
    results_dir = base_dir / results_root_name / run_name / experiment_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_best = wgan_cfg.get("save_best_model", True)
    save_every = wgan_cfg.get("save_every_n_epochs", 10)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    logger.info("Загрузка данных...")
    df = pd.read_csv(config["data_path"])
    logger.info(f"Загружено {len(df)} записей")

    logger.info("Генерация aging factor...")
    aging_generator = AgingFactorGenerator(
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
    aged = aging_generator.generate(df)
    subset = aging_generator.sample_events(
        aged,
        n_events=aging_cfg.get("n_events"),
        seed=aging_cfg.get("noise_seed"),
    )
    dataset = subset[
        [
            aging_generator.event_col,
            "cell_key",
            aging_generator.x_col,
            aging_generator.y_col,
            aging_generator.z_col,
            "E_new",
            "E_old",
            "aging_factor",
        ]
    ]
    logger.info(f"После генерации aging factor: {len(dataset)} записей")

    logger.info(f"Фильтрация по координатам: x={x_range}, y={y_range}, z={z_range}")
    dataset = filter_by_xyz(dataset, x_range, y_range, z_range)
    logger.info(f"После фильтрации: {len(dataset)} записей")

    x_vals = np.sort(dataset["x"].unique())
    y_vals = np.sort(dataset["y"].unique())
    z_vals = np.sort(dataset["z"].unique())
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    z_map = {v: i for i, v in enumerate(z_vals)}

    logger.info(f"Размерность сетки: NX={nx}, NY={ny}, NZ={nz}")
    logger.info(f"X range: [{x_vals[0]:.2f}, {x_vals[-1]:.2f}]")
    logger.info(f"Y range: [{y_vals[0]:.2f}, {y_vals[-1]:.2f}]")
    logger.info(f"Z range: [{z_vals[0]:.2f}, {z_vals[-1]:.2f}]")

    train_ds = CaloEventDataset(
        dataset,
        x_map,
        y_map,
        z_map,
        nx,
        ny,
        nz,
        event_col=aging_generator.event_col,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    logger.info(f"Количество событий: {len(train_ds)}")
    logger.info(f"Количество батчей: {len(loader)}")

    logger.info("Построение реального тензора aging factor...")
    real_w = build_real_aging_tensor(dataset, x_map, y_map, z_map, nx, ny, nz)
    real_w_np = real_w.numpy()

    generator = Generator((nz, nx, ny)).to(device)
    discriminator = Discriminator(in_dim=nz).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    logger.info("Параметры обучения:")
    logger.info(f"  - Gradient penalty: {use_gp}")
    if use_gp:
        logger.info(f"  - GP lambda: {gp_lambda}")
    else:
        logger.info(f"  - Weight clipping: {clip_value}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Epochs: {n_epochs}")
    logger.info(f"  - Learning rate: {lr}")

    history = {"global_step": [], "D_loss": [], "G_loss": [], "W_rmse": [], "W_mae": [], "W_r2": []}
    best_rmse = float("inf")
    best_epoch = 0
    best_w = None

    logger.info("=" * 60)
    logger.info("Начало обучения WGAN")
    logger.info("=" * 60)

    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar, start=1):
            e_new = batch["E_new"].to(device)
            e_old = batch["E_old"].to(device)
            mask = ((e_new != 0) & (e_old != 0)).float()
            e_new = e_new * mask
            e_old = e_old * mask

            opt_d.zero_grad()
            fake = generator(e_old).detach()
            d_real = discriminator(e_new)
            d_fake = discriminator(fake)
            d_loss = -d_real.mean() + d_fake.mean()
            if use_gp:
                gp = gradient_penalty(discriminator, e_new, fake, device)
                d_loss = d_loss + gp_lambda * gp
            d_loss.backward()
            opt_d.step()

            if not use_gp:
                for param in discriminator.parameters():
                    param.data.clamp_(-clip_value, clip_value)

            opt_g.zero_grad()
            fake = generator(e_old)
            g_loss = -discriminator(fake).mean()
            g_loss.backward()
            opt_g.step()

            with torch.no_grad():
                generator.w.clamp_(0.0, 1.0)

            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))
            pbar.set_postfix(D=np.mean(d_losses), G=np.mean(g_losses))

            if batch_idx % log_every_n_steps == 0:
                experiment.log_metrics(
                    {
                        "train/D_loss_step": float(d_loss.item()),
                        "train/G_loss_step": float(g_loss.item()),
                    },
                    step=global_step,
                    epoch=epoch,
                )
            global_step += 1

        with torch.no_grad():
            w_pred = generator.w.detach().cpu().numpy()
        w_rmse, w_mae, w_r2 = compute_metrics(w_pred, real_w_np)

        history["D_loss"].append(float(np.mean(d_losses)))
        history["G_loss"].append(float(np.mean(g_losses)))
        history["W_rmse"].append(w_rmse)
        history["W_mae"].append(w_mae)
        history["W_r2"].append(w_r2)
        history["global_step"].append(global_step)

        experiment.log_metrics(
            {
                "train/global_step": global_step,
                "train/D_loss_epoch": history["D_loss"][-1],
                "train/G_loss_epoch": history["G_loss"][-1],
                "metrics/W_rmse": w_rmse,
                "metrics/W_mae": w_mae,
                "metrics/W_r2": w_r2,
            },
            step=global_step,
            epoch=epoch,
        )

        logger.info(
            f"Epoch {epoch + 1}/{n_epochs}: D={history['D_loss'][-1]:.4f}  "
            f"G={history['G_loss'][-1]:.4f}  W_RMSE={w_rmse:.4f}  W_MAE={w_mae:.4f}  W_R2={w_r2:.4f}"
        )

        if save_best and w_rmse < best_rmse:
            best_rmse = w_rmse
            best_epoch = epoch + 1
            best_w = w_pred.copy()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": opt_g.state_dict(),
                    "optimizer_D_state_dict": opt_d.state_dict(),
                    "history": history,
                    "W_rmse": w_rmse,
                    "W_mae": w_mae,
                    "W_r2": w_r2,
                },
                models_dir / "best_model.pt",
            )
            logger.info(f"  -> Новая лучшая модель сохранена (RMSE: {w_rmse:.4f})")

        if save_every and (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": opt_g.state_dict(),
                    "optimizer_D_state_dict": opt_d.state_dict(),
                    "history": history,
                },
                models_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            )
            logger.info(f"  -> Checkpoint сохранен: checkpoint_epoch_{epoch + 1}.pt")

    logger.info("=" * 60)
    logger.info("Обучение завершено")
    logger.info(f"Лучшая модель: epoch {best_epoch}, RMSE={best_rmse:.4f}")
    logger.info("=" * 60)

    logger.info("Построение графиков...")
    save_training_curves(history, results_dir / "wgan_training_curves.png")
    logger.info("  -> Сохранены кривые обучения: wgan_training_curves.png")
    save_training_history_csv(history, results_dir / "wgan_training_history.csv")
    logger.info("  -> Сохранена история обучения: wgan_training_history.csv")

    if best_w is not None:
        pred_df = build_prediction_dataframe(best_w, real_w_np, x_vals, y_vals, z_vals)
        save_true_vs_predicted(
            pred_df,
            "a_true",
            "a_pred",
            results_dir / "wgan_true_vs_predicted.png",
            title=f"WGAN: True vs Predicted (Best epoch {best_epoch}, RMSE={best_rmse:.4f})",
        )
        logger.info("  -> Сохранен график: wgan_true_vs_predicted.png")
        save_error_histogram(
            pred_df,
            "a_true",
            "a_pred",
            results_dir / "wgan_error_histogram.png",
            title=f"WGAN: Error Distribution (Best epoch {best_epoch})",
        )
        logger.info("  -> Сохранен график: wgan_error_histogram.png")
        pred_df.to_csv(results_dir / "wgan_predictions.csv", index=False)
        logger.info("  -> Сохранены предсказания: wgan_predictions.csv")

    torch.save(
        {
            "epoch": n_epochs,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": opt_g.state_dict(),
            "optimizer_D_state_dict": opt_d.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_rmse": best_rmse,
        },
        models_dir / "final_model.pt",
    )
    logger.info("  -> Финальная модель сохранена: final_model.pt")

    experiment.end()
    return history, best_rmse, best_epoch
