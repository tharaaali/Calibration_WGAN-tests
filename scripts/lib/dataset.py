"""Утилиты подготовки датасета для обучения WGAN."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CaloEventDataset(Dataset):
    """Датасет событий калориметра для батчевого обучения."""

    def __init__(
        self,
        df: pd.DataFrame,
        x_map: dict,
        y_map: dict,
        z_map: dict,
        nx: int,
        ny: int,
        nz: int,
        event_col: str = "event",
    ):
        self.events = []
        self.x_map = x_map
        self.y_map = y_map
        self.z_map = z_map
        self.nx = nx
        self.ny = ny
        self.nz = nz

        for _, ev in df.groupby(event_col, sort=False):
            self.events.append(ev[["x", "y", "z", "E_new", "E_old"]].to_numpy())

    def __len__(self):
        """Возвращает число событий."""
        return len(self.events)

    def __getitem__(self, idx):
        """Возвращает тензоры E_new, E_old и mask для события."""
        ev = self.events[idx]
        e_new = torch.zeros(self.nz, self.nx, self.ny)
        e_old = torch.zeros(self.nz, self.nx, self.ny)
        mask = torch.zeros(self.nz, self.nx, self.ny)

        for x, y, z, en, eo in ev:
            iz = self.z_map.get(z)
            ix = self.x_map.get(x)
            iy = self.y_map.get(y)
            if iz is not None and ix is not None and iy is not None:
                e_new[iz, ix, iy] = float(en)
                e_old[iz, ix, iy] = float(eo)
                mask[iz, ix, iy] = 1.0

        return {"E_new": e_new, "E_old": e_old, "mask": mask}


class PairedCaloEventDataset(Dataset):
    """Комбинирует два датасета: из первого берёт `E_new`, из второго — `E_old`.

    Это нужно, чтобы можно было использовать независимые сэмплы событий для "new" и "old",
    но при этом сохранить интерфейс датасета для train_loop (поля `E_new` и `E_old`).
    """

    def __init__(self, ds_new: Dataset, ds_old: Dataset):
        self.ds_new = ds_new
        self.ds_old = ds_old
        self._len = min(len(ds_new), len(ds_old))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        item_new = self.ds_new[idx]
        item_old = self.ds_old[idx]
        return {"E_new": item_new["E_new"], "E_old": item_old["E_old"]}


def build_real_aging_tensor(
    df: pd.DataFrame,
    x_map: dict,
    y_map: dict,
    z_map: dict,
    nx: int,
    ny: int,
    nz: int,
) -> torch.Tensor:
    """Строит тензор истинных aging-факторов в сетке z-x-y."""
    real_w = torch.ones(nz, nx, ny)
    cell_df = (
        df[["cell_key", "x", "y", "z", "aging_factor"]]
        .drop_duplicates("cell_key")
        .reset_index(drop=True)
    )
    for _, row in cell_df.iterrows():
        x, y, z = row[["x", "y", "z"]]
        a = row["aging_factor"]
        iz = z_map.get(z)
        ix = x_map.get(x)
        iy = y_map.get(y)
        if iz is not None and ix is not None and iy is not None:
            real_w[iz, ix, iy] = float(a)
    return real_w


def build_prediction_dataframe(
    w_pred: np.ndarray,
    w_true: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
) -> pd.DataFrame:
    """Формирует таблицу предсказаний по заполненным ячейкам."""
    records = []
    nz, nx, ny = w_pred.shape
    for iz in range(nz):
        for ix in range(nx):
            for iy in range(ny):
                a_true = w_true[iz, ix, iy]
                a_pred = w_pred[iz, ix, iy]
                if abs(a_true - 1.0) < 1e-6:
                    continue
                records.append(
                    {
                        "x": x_vals[ix],
                        "y": y_vals[iy],
                        "z": z_vals[iz],
                        "a_true": a_true,
                        "a_pred": a_pred,
                    }
                )
    return pd.DataFrame(records)
