"""Основной цикл обучения WGAN."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from comet_ml import Experiment
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib.dataset import CaloEventDataset, PairedCaloEventDataset, build_prediction_dataframe, build_real_aging_tensor
from lib.models import Discriminator, Generator
from lib.plots import (
    save_error_histogram,
    save_training_curves,
    save_training_history_csv,
    save_true_vs_predicted,
)
from lib.train_utils import compute_metrics, gradient_penalty
from utils import AgingFactorGenerator, filter_by_xyz


def train_wgan(config: dict, logger: logging.Logger, experiment_name: str, run_name: str, config_name: str):
    """Запускает полный цикл обучения и сохранения артефактов."""
    model_type = str(config.get("model", "wgan")).lower()
    wgan_cfg = config.get("wgan_params", {})
    use_energy_mask = bool(wgan_cfg.get("use_energy_mask", True))
    ws_cfg = config.get("ws_params", {})
    aging_cfg = config.get("aging", {})

    use_gp = wgan_cfg.get("use_gradient_penalty", False)
    gp_lambda = wgan_cfg.get("gp_lambda", 10.0)
    clip_value = wgan_cfg.get("clip_value", 0.01)
    batch_size = wgan_cfg.get("batch_size", 64)
    n_epochs = wgan_cfg.get("epochs", 30)
    lr = wgan_cfg.get("lr", 2e-4)
    lr_g = wgan_cfg.get("lr_g")
    if lr_g is None:
        lr_g = lr
    lr_d = wgan_cfg.get("lr_d")
    if lr_d is None:
        lr_d = lr
    betas = tuple(wgan_cfg.get("betas", [0.5, 0.999]))
    n_critic = int(wgan_cfg.get("n_critic", 1))
    if n_critic < 1:
        raise ValueError(f"n_critic must be >= 1, got {n_critic}")
    disc_cfg = wgan_cfg.get("discriminator", {})
    disc_final_pool = str(disc_cfg.get("final_pool", "max"))
    seed = wgan_cfg.get("seed", 42)
    log_every_n_steps = int(wgan_cfg.get("log_every_n_steps", 100))
    use_scheduler_g = wgan_cfg.get("use_scheduler_g", False)
    use_scheduler_d = wgan_cfg.get("use_scheduler_d", False)
    scheduler_g_name = wgan_cfg.get("scheduler_g", "ReduceLROnPlateau")
    scheduler_d_name = wgan_cfg.get("scheduler_d", "ReduceLROnPlateau")
    energy_cuts_cfg = wgan_cfg.get("energy_cuts", {})
    energy_cuts_enabled = bool(energy_cuts_cfg.get("enabled", True))
    new_energy_min = float(energy_cuts_cfg.get("new_min", 200.0))
    old_energy_min = float(energy_cuts_cfg.get("old_min", 200.0))

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
            "model": model_type,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "gp_lambda": gp_lambda,
            "use_gradient_penalty": use_gp,
            "clip_value": clip_value,
            "use_scheduler_g": use_scheduler_g,
            "use_scheduler_d": use_scheduler_d,
            "scheduler_g": scheduler_g_name if use_scheduler_g else None,
            "scheduler_d": scheduler_d_name if use_scheduler_d else None,
            "lr_g": lr_g,
            "lr_d": lr_d,
            "n_critic": n_critic,
            "disc_final_pool": disc_final_pool,
        }
    )

    global_step = 0
    base_dir = Path(config.get("results_dir", "results")).parent
    logs_dir = base_dir / wgan_cfg.get("logs_dir", "logs") / run_name / experiment_name / config_name
    models_dir = base_dir / wgan_cfg.get("models_dir", "models") / run_name / experiment_name / config_name
    results_root_name = Path(config.get("results_dir", "results")).name
    results_dir = base_dir / results_root_name / run_name / experiment_name / config_name
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
        add_noise=bool(aging_cfg.get("add_noise", False)),
        sigma=aging_cfg.get("sigma", 0.0),
        noise_clamp_method=aging_cfg.get("noise_clamp_method", "clip"),
        noise_seed=aging_cfg.get("noise_seed"),
        event_col=aging_cfg.get("event_col", "event"),
        x_col=aging_cfg.get("x_col", "x"),
        y_col=aging_cfg.get("y_col", "y"),
        z_col=aging_cfg.get("z_col", "z"),
        energy_col=aging_cfg.get("energy_col", "E"),
    )
    aged = aging_generator.generate(df)
    use_test_dataset = bool(aging_cfg.get("use_test_dataset", False))
    test_fraction = float(aging_cfg.get("test_fraction", 0.2))

    independent_event_sampling = bool(aging_cfg.get("independent_event_sampling", False))
    new_events_seed = int(aging_cfg.get("new_events_seed", 1))
    old_events_seed = int(aging_cfg.get("old_events_seed", 2))
    ensure_non_overlap = bool(aging_cfg.get("ensure_non_overlap", True))

    event_ids = aged[aging_generator.event_col].drop_duplicates().to_numpy()
    n_events_total = int(aging_cfg.get("n_events"))
    n_events_total = min(n_events_total, len(event_ids))

    if independent_event_sampling:
        rng_new = np.random.default_rng(new_events_seed)
        rng_old = np.random.default_rng(old_events_seed)

        new_event_ids = rng_new.choice(event_ids, size=n_events_total, replace=False)
        if ensure_non_overlap:
            available_for_old = np.setdiff1d(event_ids, new_event_ids, assume_unique=False)
            if len(available_for_old) < n_events_total:
                raise ValueError(
                    f"Cannot sample {n_events_total} non-overlapping old events. "
                    f"Available={len(available_for_old)}, total={len(event_ids)}"
                )
            old_event_ids = rng_old.choice(available_for_old, size=n_events_total, replace=False)
        else:
            old_event_ids = rng_old.choice(event_ids, size=n_events_total, replace=False)

        if use_test_dataset and n_events_total > 1 and test_fraction > 0.0:
            test_size = int(round(n_events_total * test_fraction))
            test_size = max(1, min(test_size, n_events_total - 1))

            test_new_event_ids = new_event_ids[:test_size]
            train_new_event_ids = new_event_ids[test_size:]

            test_old_event_ids = old_event_ids[:test_size]
            train_old_event_ids = old_event_ids[test_size:]
        else:
            test_new_event_ids = None
            train_new_event_ids = new_event_ids
            test_old_event_ids = None
            train_old_event_ids = old_event_ids

        subset_train_new = aged[aged[aging_generator.event_col].isin(train_new_event_ids)].copy()
        subset_train_old = aged[aged[aging_generator.event_col].isin(train_old_event_ids)].copy()

        dataset_train = subset_train_new[
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

        dataset_train_old = subset_train_old[
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

        dataset_test = None
        dataset_test_old = None
        if test_new_event_ids is not None:
            subset_test_new = aged[aged[aging_generator.event_col].isin(test_new_event_ids)].copy()
            subset_test_old = aged[aged[aging_generator.event_col].isin(test_old_event_ids)].copy()

            dataset_test = subset_test_new[
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
            dataset_test_old = subset_test_old[
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
    else:
        noise_seed = aging_cfg.get("noise_seed", 42)
        rng = np.random.default_rng(int(noise_seed))
        sampled_event_ids = rng.choice(event_ids, size=n_events_total, replace=False)

        if use_test_dataset and n_events_total > 1 and test_fraction > 0.0:
            test_size = int(round(n_events_total * test_fraction))
            test_size = max(1, min(test_size, n_events_total - 1))
            test_event_ids = sampled_event_ids[:test_size]
            train_event_ids = sampled_event_ids[test_size:]
        else:
            test_event_ids = None
            train_event_ids = sampled_event_ids

        subset_train = aged[aged[aging_generator.event_col].isin(train_event_ids)].copy()
        dataset_train = subset_train[
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
        dataset_train_old = dataset_train

        dataset_test = None
        dataset_test_old = None
        if test_event_ids is not None:
            subset_test = aged[aged[aging_generator.event_col].isin(test_event_ids)].copy()
            dataset_test = subset_test[
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
            dataset_test_old = dataset_test

    logger.info(
        "После генерации aging factor: train records=%d, test records=%s",
        len(dataset_train),
        "None" if dataset_test is None else len(dataset_test),
    )

    logger.info(f"Фильтрация по координатам: x={x_range}, y={y_range}, z={z_range}")
    if energy_cuts_enabled:
        train_new_before = len(dataset_train)
        train_old_before = len(dataset_train_old)
        dataset_train = dataset_train[dataset_train["E_new"] >= new_energy_min].copy()
        dataset_train_old = dataset_train_old[dataset_train_old["E_old"] >= old_energy_min].copy()

        if dataset_test is not None:
            dataset_test = dataset_test[dataset_test["E_new"] >= new_energy_min].copy()
        if dataset_test_old is not None:
            dataset_test_old = dataset_test_old[dataset_test_old["E_old"] >= old_energy_min].copy()

        logger.info(
            "Energy cuts: E_new >= %.1f, E_old >= %.1f | train_new: %d -> %d, train_old: %d -> %d",
            new_energy_min,
            old_energy_min,
            train_new_before,
            len(dataset_train),
            train_old_before,
            len(dataset_train_old),
        )

    dataset_train = filter_by_xyz(dataset_train, x_range, y_range, z_range)
    dataset_train_old = filter_by_xyz(dataset_train_old, x_range, y_range, z_range)

    dataset_test_filtered = None
    dataset_test_old_filtered = None
    if dataset_test is not None:
        dataset_test_filtered = filter_by_xyz(dataset_test, x_range, y_range, z_range)
    if dataset_test_old is not None:
        dataset_test_old_filtered = filter_by_xyz(dataset_test_old, x_range, y_range, z_range)

    logger.info(
        "После фильтрации: train_new records=%d, train_old records=%d, test_new=%s",
        len(dataset_train),
        len(dataset_train_old),
        "None" if dataset_test_filtered is None else len(dataset_test_filtered),
    )

    x_vals = np.sort(dataset_train["x"].unique())
    y_vals = np.sort(dataset_train["y"].unique())
    z_vals = np.sort(dataset_train["z"].unique())
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    z_map = {v: i for i, v in enumerate(z_vals)}

    logger.info(f"Размерность сетки: NX={nx}, NY={ny}, NZ={nz}")
    logger.info(f"X range: [{x_vals[0]:.2f}, {x_vals[-1]:.2f}]")
    logger.info(f"Y range: [{y_vals[0]:.2f}, {y_vals[-1]:.2f}]")
    logger.info(f"Z range: [{z_vals[0]:.2f}, {z_vals[-1]:.2f}]")

    if model_type == "ws":
        logger.info("Режим model=ws: вычисление коэффициентов через Wasserstein distance")

        dataset_quality_new = dataset_test_filtered if use_test_dataset else dataset_train
        dataset_quality_old = dataset_test_old_filtered if use_test_dataset else dataset_train_old
        if dataset_quality_new is None or len(dataset_quality_new) == 0:
            dataset_quality_new = dataset_train
        if dataset_quality_old is None or len(dataset_quality_old) == 0:
            dataset_quality_old = dataset_train_old

        real_w = build_real_aging_tensor(dataset_quality_new, x_map, y_map, z_map, nx, ny, nz)
        real_w_np = real_w.numpy()

        alpha_min = float(ws_cfg.get("alpha_min", 0.80))
        alpha_max = float(ws_cfg.get("alpha_max", 1.00))
        alpha_step = float(ws_cfg.get("alpha_step", 0.01))
        alpha_grid = np.arange(alpha_min, alpha_max + 0.5 * alpha_step, alpha_step)

        group_new = dataset_quality_new.groupby("cell_key")["E_new"]
        group_old = dataset_quality_old.groupby("cell_key")["E_old"]
        common_cells = sorted(set(group_new.groups.keys()) & set(group_old.groups.keys()))
        if not common_cells:
            raise ValueError("No common cells between E_new and E_old datasets for WS model")

        a_hat_wasserstein = {}
        for cell in common_cells:
            x_new = group_new.get_group(cell).to_numpy()
            x_old = group_old.get_group(cell).to_numpy()

            w_vals = [wasserstein_distance(x_new, x_old / alpha) for alpha in alpha_grid]
            best_alpha = float(alpha_grid[int(np.argmin(w_vals))])
            a_hat_wasserstein[cell] = best_alpha

        ws_df = pd.DataFrame.from_dict(a_hat_wasserstein, orient="index", columns=["a_pred"])
        ws_df.index.name = "cell_key"

        cell_coords = (
            dataset_quality_new[["cell_key", "x", "y", "z"]]
            .drop_duplicates("cell_key")
            .set_index("cell_key")
        )
        ws_df = ws_df.join(cell_coords, how="inner").dropna()

        w_pred = np.ones((nz, nx, ny), dtype=np.float32)
        for _, row in ws_df.iterrows():
            x = row["x"]
            y = row["y"]
            z = row["z"]
            if x in x_map and y in y_map and z in z_map:
                w_pred[z_map[z], x_map[x], y_map[y]] = float(row["a_pred"])

        w_rmse, w_mae, w_r2 = compute_metrics(w_pred, real_w_np)
        history = {
            "global_step": [0],
            "D_loss": [float("nan")],
            "G_loss": [float("nan")],
            "W_rmse": [w_rmse],
            "W_mae": [w_mae],
            "W_r2": [w_r2],
        }
        best_rmse = w_rmse
        best_epoch = 1
        best_w = w_pred

        logger.info(f"WS metrics: W_RMSE={w_rmse:.4f}  W_MAE={w_mae:.4f}  W_R2={w_r2:.4f}")
        experiment.log_metrics(
            {
                "train/global_step": 0,
                "metrics/W_rmse": w_rmse,
                "metrics/W_mae": w_mae,
                "metrics/W_r2": w_r2,
            },
            step=0,
            epoch=0,
        )

        save_training_curves(history, results_dir / "wgan_training_curves.png")
        save_training_history_csv(history, results_dir / "wgan_training_history.csv")

        pred_df = build_prediction_dataframe(best_w, real_w_np, x_vals, y_vals, z_vals)
        save_true_vs_predicted(
            pred_df,
            "a_true",
            "a_pred",
            results_dir / "wgan_true_vs_predicted.png",
            title=f"WS: True vs Predicted (RMSE={best_rmse:.4f})",
        )
        save_error_histogram(
            pred_df,
            "a_true",
            "a_pred",
            results_dir / "wgan_error_histogram.png",
            title="WS: Error Distribution",
        )
        pred_df.to_csv(results_dir / "wgan_predictions.csv", index=False)
        ws_df.reset_index().to_csv(models_dir / "ws_cell_alphas.csv", index=False)

        experiment.end()
        return history, best_rmse, best_epoch
    if independent_event_sampling:
        train_ds_new = CaloEventDataset(
            dataset_train,
            x_map,
            y_map,
            z_map,
            nx,
            ny,
            nz,
            event_col=aging_generator.event_col,
        )
        train_ds_old = CaloEventDataset(
            dataset_train_old,
            x_map,
            y_map,
            z_map,
            nx,
            ny,
            nz,
            event_col=aging_generator.event_col,
        )
        train_ds = PairedCaloEventDataset(train_ds_new, train_ds_old)
    else:
        train_ds = CaloEventDataset(
            dataset_train,
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
    dataset_quality = dataset_test_filtered if use_test_dataset else dataset_train
    if dataset_quality is None or len(dataset_quality) == 0:
        dataset_quality = dataset_train
    real_w = build_real_aging_tensor(dataset_quality, x_map, y_map, z_map, nx, ny, nz)
    real_w_np = real_w.numpy()

    generator = Generator((nz, nx, ny)).to(device)
    discriminator = Discriminator(in_dim=nz, final_pool=disc_final_pool).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    scheduler_g = None
    scheduler_d = None
    if use_scheduler_g:
        if scheduler_g_name == "ReduceLROnPlateau":
            scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.5, patience=5)
        elif scheduler_g_name == "StepLR":
            scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=10, gamma=0.5)
        elif scheduler_g_name == "CosineAnnealingLR":
            scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=n_epochs)
    if use_scheduler_d:
        if scheduler_d_name == "ReduceLROnPlateau":
            scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.5, patience=5)
        elif scheduler_d_name == "StepLR":
            scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=10, gamma=0.5)
        elif scheduler_d_name == "CosineAnnealingLR":
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=n_epochs)

    logger.info("Параметры обучения:")
    logger.info(f"  - Gradient penalty: {use_gp}")
    if use_gp:
        logger.info(f"  - GP lambda: {gp_lambda}")
    else:
        logger.info(f"  - Weight clipping: {clip_value}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Epochs: {n_epochs}")
    logger.info(f"  - Learning rate G: {lr_g}")
    logger.info(f"  - Learning rate D: {lr_d}")
    logger.info(f"  - n_critic: {n_critic}")
    logger.info(f"  - Discriminator final pool: {disc_final_pool}")
    logger.info(f"  - Scheduler G: {scheduler_g_name if use_scheduler_g else 'None'}")
    logger.info(f"  - Scheduler D: {scheduler_d_name if use_scheduler_d else 'None'}")

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
            if use_energy_mask:
                mask = (e_old != 0).float()
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

            d_losses.append(float(d_loss.item()))

            do_g_step = (batch_idx % n_critic == 0)
            if do_g_step:
                opt_g.zero_grad()
                fake = generator(e_old)
                g_loss = -discriminator(fake).mean()
                g_loss.backward()
                opt_g.step()

                with torch.no_grad():
                    generator.w.clamp_(0.0, 1.0)

                g_losses.append(float(g_loss.item()))

            g_postfix = float(np.mean(g_losses)) if g_losses else float("nan")
            pbar.set_postfix(D=np.mean(d_losses), G=g_postfix)

            if batch_idx % log_every_n_steps == 0:
                metrics_to_log = {"train/D_loss_step": float(d_loss.item())}
                if do_g_step:
                    metrics_to_log["train/G_loss_step"] = float(g_loss.item())
                experiment.log_metrics(
                    metrics_to_log,
                    step=global_step,
                    epoch=epoch,
                )
            global_step += 1

        with torch.no_grad():
            w_pred = generator.w.detach().cpu().numpy()
        w_rmse, w_mae, w_r2 = compute_metrics(w_pred, real_w_np)

        history["D_loss"].append(float(np.mean(d_losses)) if d_losses else float("nan"))
        history["G_loss"].append(float(np.mean(g_losses)) if g_losses else float("nan"))
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

        if scheduler_g:
            if scheduler_g_name == "ReduceLROnPlateau":
                scheduler_g.step(w_rmse)
            else:
                scheduler_g.step()
        if scheduler_d:
            if scheduler_d_name == "ReduceLROnPlateau":
                scheduler_d.step(w_rmse)
            else:
                scheduler_d.step()

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
