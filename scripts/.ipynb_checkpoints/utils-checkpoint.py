from typing import Optional, Tuple
import numpy as np
import pandas as pd


class AgingFactorGenerator:
    """Генератор коэффициентов старения для ячеек калориметра."""
    def __init__(
        self,
        *,
        aging_function: str = "xyz",
        af_min: float = 0.8,
        af_max: float = 1.0,
        sigma: float = 0.0,
        noise_seed: Optional[int] = None,
        event_col: str = "event",
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        energy_col: str = "E",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        zlim: Optional[Tuple[float, float]] = None,
        alpha: float = 1.6,
        kz: float = 0.08,
        lambda_z: float = 0.20,
        eta: float = 2.0
    ) -> None:
        """Инициализация параметров генерации AF и выборки событий."""
        self.aging_function = aging_function
        self.af_min = af_min
        self.af_max = af_max
        self.sigma = sigma
        self.noise_seed = noise_seed
        self.event_col = event_col
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.energy_col = energy_col
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.alpha = alpha
        self.kz = kz
        self.lambda_z = lambda_z
        self.eta = eta
        self._methods = {
            "xyz": self._base_xyz,
            "rz": self._base_rz,
            "frequency": self._base_frequency,
        }

    def _cell_index(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> pd.Series:
        """Формирует идентификатор ячейки как кортеж координат."""
        return pd.Series(list(zip(df[x_col].to_numpy(), df[y_col].to_numpy(), df[z_col].to_numpy())), index=df.index)

    def _unit(self, series: pd.Series) -> pd.Series:
        """Линейно нормализует значения в диапазон [0,1]."""
        vmin = series.min()
        vmax = series.max()
        if vmax == vmin:
            return pd.Series(0.5, index=series.index, dtype=float)
        return (series - vmin) / (vmax - vmin)

    def _unique_geom(self, df: pd.DataFrame, key: pd.Series, x_col: str, y_col: str, z_col: str) -> pd.DataFrame:
        """Получает уникальную геометрию ячеек."""
        geom_df = pd.DataFrame({
            'cell_key': key,
            x_col: df[x_col],
            y_col: df[y_col],
            z_col: df[z_col]
        }).drop_duplicates(subset=['cell_key'], keep='first').set_index('cell_key')
        return geom_df

    def _limits_or_infer(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Определяет границы координат из данных или использует заданные."""
        xlim = self.xlim if self.xlim is not None else (float(df[x_col].min()), float(df[x_col].max()))
        ylim = self.ylim if self.ylim is not None else (float(df[y_col].min()), float(df[y_col].max()))
        zlim = self.zlim if self.zlim is not None else (float(df[z_col].min()), float(df[z_col].max()))
        return xlim, ylim, zlim

    def _base_xyz(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, key: pd.Series) -> pd.Series:
        """Возвращает базовый AF на основе координат x,y,z (как в референсе)."""
        xlim, ylim, zlim = self._limits_or_infer(df, x_col, y_col, z_col)
        geom = self._unique_geom(df, key, x_col, y_col, z_col)
        
        (x1, x2), (y1, y2), (z1, z2) = xlim, ylim, zlim
        cx, rxh = (x1 + x2) / 2.0, max(1e-12, (x2 - x1) / 2.0)
        cy, ryh = (y1 + y2) / 2.0, max(1e-12, (y2 - y1) / 2.0)
        cz, rzh = (z1 + z2) / 2.0, max(1e-12, (z2 - z1) / 2.0)

        x = geom[x_col].to_numpy(copy=False)
        y = geom[y_col].to_numpy(copy=False)
        z = geom[z_col].to_numpy(copy=False)

        dx = (x - cx) / rxh
        dy = (y - cy) / ryh
        r = np.sqrt(dx*dx + dy*dy)
        np.clip(r, 0.0, 1.0, out=r)

        zhat = (z - cz) / rzh
        g = (r ** self.alpha) + (0.5 * self.kz) * zhat

        fac = g.copy()
        np.clip(fac, 0.0, 1.0, out=fac)

        mapping = pd.Series(fac, index=geom.index)
        return key.map(mapping)

    def _base_rz(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, key: pd.Series) -> pd.Series:
        """Возвращает базовый AF в координатах r,z (как в референсе)."""
        xlim, ylim, zlim = self._limits_or_infer(df, x_col, y_col, z_col)
        geom = self._unique_geom(df, key, x_col, y_col, z_col)
        
        (x1, x2), (y1, y2), (z1, z2) = xlim, ylim, zlim
        cx, rxh = (x1 + x2) / 2.0, max(1e-12, (x2 - x1) / 2.0)
        cy, ryh = (y1 + y2) / 2.0, max(1e-12, (y2 - y1) / 2.0)
        cz, rzh = (z1 + z2) / 2.0, max(1e-12, (z2 - z1) / 2.0)

        x = geom[x_col].to_numpy(copy=False)
        y = geom[y_col].to_numpy(copy=False)
        z = geom[z_col].to_numpy(copy=False)

        dx = (x - cx) / rxh
        dy = (y - cy) / ryh
        r = np.sqrt(dx*dx + dy*dy)
        np.clip(r, 0.0, 1.0, out=r)

        zhat = (z - cz) / rzh
        zshape = np.abs(zhat)
        if self.eta != 1.0:
            zshape = zshape ** self.eta

        g = (1.0 - self.lambda_z) * (r ** self.alpha) + self.lambda_z * zshape
        np.clip(g, 0.0, 1.0, out=g)

        mapping = pd.Series(g, index=geom.index)
        return key.map(mapping)

    def _base_frequency(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str, key: pd.Series) -> pd.Series:
        """Возвращает базовый AF в зависимости от частоты появления ячейки (как в референсе)."""
        freq = key.value_counts().sort_values(ascending=True)
        ranks = freq.rank(method="dense", ascending=True)
        K = int(ranks.max())
        if K == 1:
            scale = pd.Series(1.0, index=freq.index)
        else:
            scale = 1.0 - (ranks - 1) / (K - 1)
        mapping = scale.to_dict()
        return key.map(mapping)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерирует коэффициенты старения и энергии E_new/E_old."""
        x_col = self.x_col
        y_col = self.y_col
        z_col = self.z_col
        e_col = self.energy_col
        key = self._cell_index(df, x_col, y_col, z_col)
        method = self._methods.get(self.aging_function, self._base_xyz)
        base_unit = method(df, x_col, y_col, z_col, key)
        base = self.af_min + (self.af_max - self.af_min) * base_unit.clip(0.0, 1.0)
        rng = np.random.default_rng(self.noise_seed)
        af = np.clip(base + self.sigma * rng.normal(size=len(df)), self.af_min, self.af_max)
        out = df.copy()
        out["cell_key"] = key
        out["aging_factor"] = af
        out["E_new"] = out[e_col]
        out["E_old"] = out["E_new"] * out["aging_factor"]
        # mask_old = out["E_old"] < 200
        # out.loc[mask_old, "E_old"] = np.nan
        return out

    def sample_events(self, df: pd.DataFrame, *, n_events: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
        """Возвращает подвыборку по случайным событиям."""
        if n_events is None:
            return df
        unique_events = df[self.event_col].drop_duplicates()
        if n_events >= len(unique_events):
            return df
        sampled = unique_events.sample(n=n_events, random_state=seed)
        return df[df[self.event_col].isin(sampled)].copy()


def aggregate_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует статистику по ячейкам."""
    cell_stats = df.dropna(subset=["E_new", "E_old"]).groupby("cell_key").agg(
        freq=("cell_key", "size"),
        x=("x", "first"),
        y=("y", "first"),
        z=("z", "first"),
        aging=("aging_factor", "mean"),
        mean_E_new=("E_new", "mean"),
        mean_E_old=("E_old", "mean"),
    )
    cell_stats["shift"] = cell_stats["mean_E_old"] - cell_stats["mean_E_new"]
    cell_stats["ratio"] = cell_stats["mean_E_old"] / cell_stats["mean_E_new"]
    return cell_stats


def get_cellid_counts_per_event(df: pd.DataFrame, event_col: str = "event") -> np.ndarray:
    """Возвращает массив количества уникальных cell_key на событие."""
    return df.groupby(event_col)["cell_key"].nunique().to_numpy()


def filter_by_xyz(df: pd.DataFrame, x_range: Optional[Tuple[float, float]] = None, 
                  y_range: Optional[Tuple[float, float]] = None, 
                  z_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Фильтрует DataFrame по диапазонам координат.
    Пример: x_range=(-30, 30), y_range=(-30, 30), z_range=(-100, -80)
    """
    mask = pd.Series(True, index=df.index)

    if x_range is not None:
        xmin, xmax = x_range
        mask &= df["x"].between(xmin, xmax)

    if y_range is not None:
        ymin, ymax = y_range
        mask &= df["y"].between(ymin, ymax)

    if z_range is not None:
        zmin, zmax = z_range
        mask &= df["z"].between(zmin, zmax)

    return df[mask].copy()
