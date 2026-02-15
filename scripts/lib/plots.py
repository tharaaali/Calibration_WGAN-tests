"""Функции построения и сохранения графиков WGAN."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def save_training_curves(history: dict, save_path: Path):
    """Сохраняет кривые потерь и метрик по эпохам."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = range(1, len(history["D_loss"]) + 1)

    axes[0, 0].plot(epochs, history["D_loss"], "b-", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Discriminator Loss")
    axes[0, 0].set_title("Discriminator Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["G_loss"], "r-", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Generator Loss")
    axes[0, 1].set_title("Generator Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history["W_rmse"], "g-", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("RMSE")
    axes[0, 2].set_title("Weight RMSE")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["W_mae"], "m-", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].set_title("Weight MAE")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["W_r2"], "c-", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("R2")
    axes[1, 1].set_title("Weight R2")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_history_csv(history: dict, save_path: Path):
    """Сохраняет историю обучения в CSV."""
    epochs = range(1, len(history["D_loss"]) + 1)
    data = {
        "epoch": list(epochs),
        "D_loss": history["D_loss"],
        "G_loss": history["G_loss"],
        "W_rmse": history["W_rmse"],
        "W_mae": history["W_mae"],
        "W_r2": history["W_r2"],
    }
    if "global_step" in history:
        data["global_step"] = history["global_step"]
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def save_true_vs_predicted(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    save_path: Path,
    title: str = "True vs Predicted",
):
    """Сохраняет scatter истинных и предсказанных значений."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df[true_col], df[pred_col], s=8, alpha=0.5)

    lims = [
        min(df[true_col].min(), df[pred_col].min()),
        max(df[true_col].max(), df[pred_col].max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Ideal")
    ax.set_xlabel(r"True aging factor $a_i$", fontsize=12)
    ax.set_ylabel(r"Predicted $\hat{a}_i$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_error_histogram(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    save_path: Path,
    title: str = "Error Distribution",
):
    """Сохраняет гистограмму ошибки предсказания."""
    errors = df[true_col] - df[pred_col]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=60, alpha=0.8, edgecolor="black", linewidth=1.2)

    mean_err = errors.mean()
    std_err = errors.std()
    ax.axvline(
        mean_err,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_err:.4f}",
    )
    ax.axvline(
        mean_err + std_err,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Std: {std_err:.4f}",
    )
    ax.axvline(mean_err - std_err, color="orange", linestyle=":", linewidth=2)
    ax.set_xlabel(r"Error: $a_i - \hat{a}_i$", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
