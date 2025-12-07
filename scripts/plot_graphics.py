from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np


class PlotGenerator:
    """Генератор графиков для анализа данных калориметра."""
    
    def __init__(self, config: dict):
        """Инициализация генератора графиков."""
        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._methods = {
            "energy_distribution": self._plot_energy_distribution,
            "aging_vs_frequency": self._plot_aging_vs_frequency,
            "aging_correlations": self._plot_aging_correlations,
            "global_shift_hist": self._plot_global_shift_hist,
            "global_energy_shift": self._plot_global_energy_shift,
        }
    
    def plot(self, dataset: pd.DataFrame, plot_type: str, config: dict):
        """Построение графика указанного типа."""
        plot_config = config["plots"][plot_type]
        return self._methods[plot_type](dataset, **plot_config)
    
    def plot_aging_vs_frequency(self, cell_stats: pd.DataFrame, plot_type: str, config: dict):
        """Построение зависимости aging-фактора от частоты."""
        plot_config = config["plots"][plot_type]
        return self._plot_aging_vs_frequency(cell_stats, **plot_config)
    
    def plot_aging_correlations(self, cell_stats: pd.DataFrame, plot_type: str, config: dict):
        """Построение корреляционной матрицы."""
        plot_config = config["plots"][plot_type].copy()
        if "figsize" in plot_config and isinstance(plot_config["figsize"], list):
            plot_config["figsize"] = tuple(plot_config["figsize"])
        return self._plot_aging_correlations(cell_stats, **plot_config)
    
    def plot_global_shift_hist(self, cell_stats: pd.DataFrame, plot_type: str, config: dict):
        """Построение распределения E_old - E_new."""
        plot_config = config["plots"][plot_type]
        return self._plot_global_shift_hist(cell_stats, **plot_config)
    
    def plot_global_energy_shift(self, dataset: pd.DataFrame, plot_type: str, config: dict):
        """Построение scatter plot E_new vs E_old."""
        plot_config = config["plots"][plot_type]
        return self._plot_global_energy_shift(dataset, **plot_config)
    
    def plot_energy_distribution_cell(self, dataset: pd.DataFrame, cell_stats: pd.DataFrame, plot_type: str, config: dict):
        """Построение распределения энергии для выбранной ячейки."""
        plot_config = config["plots"][plot_type].copy()
        cell_idx = plot_config.pop("cell_idx", None)
        if cell_idx is None:
            cell_idx = cell_stats['freq'].idxmax()
            cell_id = cell_idx
        else:
            cell_id = cell_stats.index[cell_idx]
        plot_config["cell_id"] = cell_id
        plot_config["cell_idx"] = cell_idx
        return self._plot_energy_distribution_cell(dataset, cell_stats, **plot_config)
    
    def plot_spatial_distribution(self, df: pd.DataFrame, mode: str, filename: str, config: dict):
        """Построение 2D гистограммы пространственного распределения."""
        plot_config = config["plots"]["spatial_distribution"]
        return self._plot_spatial_distribution(df, mode, filename, **plot_config)
    
    def plot_cell_activity_distribution(self, cellid_counts: np.ndarray, plot_type: str, config: dict):
        """Построение распределения количества уникальных cell_key на событие."""
        plot_config = config["plots"][plot_type]
        return self._plot_cell_activity_distribution(cellid_counts, **plot_config)
    
    def plot_aging_factor_xy(self, dataset: pd.DataFrame, plot_type: str, config: dict):
        """Построение зависимости aging factor от X-Y для фиксированного Z."""
        plot_config = config["plots"][plot_type]
        return self._plot_aging_factor_xy(dataset, **plot_config)
    
    def plot_aging_factor_z(self, dataset: pd.DataFrame, plot_type: str, config: dict):
        """Построение зависимости aging factor от Z для фиксированных X и Y."""
        plot_config = config["plots"][plot_type]
        return self._plot_aging_factor_z(dataset, **plot_config)
    
    def plot_energy_histogram_cell(self, dataset: pd.DataFrame, cell_stats: pd.DataFrame, plot_type: str, config: dict):
        """Построение гистограмм распределения энергии для выбранной ячейки."""
        plot_config = config["plots"][plot_type].copy()
        cell_key = plot_config.pop("cell_key", None)
        if cell_key is None:
            freq_threshold = cell_stats['freq'].quantile(0.75)
            high_freq_cells = cell_stats[cell_stats['freq'] >= freq_threshold]
            if len(high_freq_cells) > 0:
                cell_id = high_freq_cells.sample(1).index[0]
            else:
                cell_id = cell_stats['freq'].idxmax()
        else:
            if isinstance(cell_key, list):
                cell_key = tuple(cell_key)
            cell_id = cell_key
            if cell_id not in cell_stats.index:
                freq_threshold = cell_stats['freq'].quantile(0.75)
                high_freq_cells = cell_stats[cell_stats['freq'] >= freq_threshold]
                if len(high_freq_cells) > 0:
                    cell_id = high_freq_cells.sample(1).index[0]
                else:
                    cell_id = cell_stats['freq'].idxmax()
        plot_config["cell_id"] = cell_id
        return self._plot_energy_histogram_cell(dataset, cell_stats, **plot_config)
    
    def _plot_energy_distribution(
        self,
        dataset: pd.DataFrame,
        e_new_col: str = "E_new",
        e_old_col: str = "E_old",
        quantile: float = 0.95,
        bins: int = 50,
        filename: str = "energy_distribution.png",
        dpi: int = 150
    ):
        """Построение распределения энергий E_new и E_old."""
        energy_vals = dataset[[e_new_col, e_old_col]].dropna()
        upper_quantile = energy_vals.quantile(quantile).max()
        
        fig, ax = plt.subplots()
        sns.histplot(
            energy_vals[e_new_col],
            ax=ax,
            color="#1f77b4",
            stat="density",
            kde=False,
            label="E_new",
            bins=bins
        )
        sns.histplot(
            energy_vals[e_old_col],
            ax=ax,
            color="#ff7f0e",
            stat="density",
            kde=False,
            label="E_old",
            bins=bins
        )
        ax.set_xlabel("Energy")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(left=0.0, right=upper_quantile)
        
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_aging_vs_frequency(
        self,
        cell_stats: pd.DataFrame,
        xlim: tuple = None,
        filename: str = "aging_vs_frequency.png",
        dpi: int = 150
    ):
        """Построение зависимости aging-фактора от частоты ячейки."""
        freq_profile = cell_stats.groupby("freq")["aging"].mean().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(data=freq_profile, x="freq", y="aging", ax=ax, marker="o")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Mean aging factor")
        if xlim:
            ax.set_xlim(xlim)
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_aging_correlations(
        self,
        cell_stats: pd.DataFrame,
        columns: list,
        filename: str = "aging_correlations.png",
        figsize: tuple = (7, 5),
        dpi: int = 150
    ):
        """Построение корреляционной матрицы."""
        corr = cell_stats[columns].corr()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f", cbar_kws={"shrink": 0.8})
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_global_shift_hist(
        self,
        cell_stats: pd.DataFrame,
        quantile: float = 0.95,
        kde: bool = True,
        filename: str = "global_shift_hist.png",
        dpi: int = 150
    ):
        """Построение распределения E_old - E_new."""
        shift_vals = cell_stats["shift"].dropna()
        limit = shift_vals.abs().quantile(quantile)
        fig, ax = plt.subplots()
        sns.histplot(shift_vals, ax=ax, kde=kde, color="#2ca02c", stat="density")
        ax.set_xlabel("E_old - E_new")
        ax.set_ylabel("Density")
        ax.set_xlim(-limit, limit)
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_global_energy_shift(
        self,
        dataset: pd.DataFrame,
        filename: str = "global_energy_shift.png",
        dpi: int = 150
    ):
        """Построение scatter plot E_new vs E_old."""
        energy_quantile = dataset[["E_new", "E_old"]].quantile(0.95)
        upper_limit = float(max(energy_quantile["E_new"], energy_quantile["E_old"]))
        fig, ax = plt.subplots()
        sns.scatterplot(data=dataset, x="E_new", y="E_old", ax=ax, s=5, alpha=0.3)
        ax.plot([0.0, upper_limit], [0.0, upper_limit], color="black", linewidth=1)
        ax.set_xlim(0.0, upper_limit)
        ax.set_ylim(0.0, upper_limit)
        ax.set_xlabel("E_new")
        ax.set_ylabel("E_old")
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_energy_distribution_cell(
        self,
        dataset: pd.DataFrame,
        cell_stats: pd.DataFrame,
        cell_id,
        cell_idx: int,
        quantile: float = 0.99,
        filename: str = "energy_distribution_cell.png",
        dpi: int = 150
    ):
        """Построение распределения энергии для выбранной ячейки."""
        cell_df = dataset[dataset['cell_key'] == cell_id]
        if cell_df.empty:
            raise ValueError(f"Ячейка с cell_id={cell_id} не найдена в данных")
        cell_quantile = cell_df[["E_new", "E_old"]].dropna().quantile(quantile).max()
        if pd.isna(cell_quantile) or cell_quantile == 0:
            cell_quantile = cell_df[["E_new", "E_old"]].max().max()
        fig, ax = plt.subplots()
        sns.histplot(cell_df["E_new"], ax=ax, color="#1f77b4", stat="density", kde=False, label="E_new")
        sns.histplot(cell_df["E_old"], ax=ax, color="#ff7f0e", stat="density", kde=False, label="E_old")
        ax.set_title(f"Cell {cell_id}")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Density")
        ax.legend()
        if not pd.isna(cell_quantile) and cell_quantile > 0:
            ax.set_xlim(left=0.0, right=cell_quantile)
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_spatial_distribution(
        self,
        df: pd.DataFrame,
        mode: str,
        filename: str,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
        z_min: float = None,
        z_max: float = None,
        dpi: int = 150
    ):
        """Построение 2D гистограммы пространственного распределения."""
        if mode == "xy":
            x_col, y_col = "x", "y"
            x_lim = (x_min, x_max) if x_min is not None and x_max is not None else None
            y_lim = (y_min, y_max) if y_min is not None and y_max is not None else None
            bins = [48, 48]
            figsize = (6, 6)
            title = "Spatial distribution of energy X-Y"
            xlabel, ylabel = "X", "Y"
        elif mode == "zy":
            x_col, y_col = "z", "y"
            x_lim = (z_min, z_max) if z_min is not None and z_max is not None else None
            y_lim = (y_min, y_max) if y_min is not None and y_max is not None else None
            bins = [40, 48]
            figsize = (6, 8)
            title = "Spatial distribution of energy Z-Y"
            xlabel, ylabel = "Z", "Y"
        elif mode == "zx":
            x_col, y_col = "z", "x"
            x_lim = (z_min, z_max) if z_min is not None and z_max is not None else None
            y_lim = (x_min, x_max) if x_min is not None and x_max is not None else None
            bins = [40, 48]
            figsize = (6, 8)
            title = "Spatial distribution of energy Z-X"
            xlabel, ylabel = "Z", "X"
        else:
            raise ValueError(f"Неизвестный режим: {mode}. Доступны: xy, zy, zx")
        
        filtered_df = df.copy()
        if x_lim:
            filtered_df = filtered_df[(filtered_df[x_col] >= x_lim[0]) & (filtered_df[x_col] <= x_lim[1])]
        if y_lim:
            filtered_df = filtered_df[(filtered_df[y_col] >= y_lim[0]) & (filtered_df[y_col] <= y_lim[1])]
        
        with plt.style.context('default'):
            fig = plt.figure(figsize=figsize)
            hist, xedges, yedges, img = plt.hist2d(filtered_df[x_col], filtered_df[y_col], bins=bins, cmap='viridis')
            plt.grid(False, which='both')
            ax = plt.gca()
            ax.grid(False)
            cbar = plt.colorbar(img)
            cbar.set_label('Counts')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if x_lim:
                plt.xlim(x_lim)
            if y_lim:
                plt.ylim(y_lim)
            fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_cell_activity_distribution(
        self,
        cellid_counts: np.ndarray,
        figsize: tuple = (6, 4),
        bins: int = 50,
        filename: str = "distribution_of_cell_activity_per_event.png",
        dpi: int = 150
    ):
        """Построение распределения количества уникальных cell_key на событие."""
        if isinstance(figsize, list):
            figsize = tuple(figsize)
        fig = plt.figure(figsize=figsize)
        plt.hist(cellid_counts, bins=bins, edgecolor="black", linewidth=1.2)
        plt.xlabel("Number of Unique Cell IDs per Event", fontweight="bold")
        plt.ylabel("Event Count", fontweight="bold")
        plt.title("Distribution of Cell Activity per Event", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        return fig
    
    def _plot_aging_factor_xy(
        self,
        dataset: pd.DataFrame,
        z_fixed: float = -90.0,
        n_events: int = 1000,
        filename: str = "aging_factor_xy.png",
        dpi: int = 150
    ):
        """Построение зависимости aging factor от X-Y для фиксированного Z."""
        event_counts = dataset.groupby("event")["cell_key"].nunique()
        events_with_multiple_cells = event_counts[event_counts > 1].index
        selected_events = np.random.choice(
            events_with_multiple_cells,
            size=min(n_events, len(events_with_multiple_cells)),
            replace=False
        )
        df_multi = dataset[dataset["event"].isin(selected_events)]
        
        available_z = df_multi["z"].unique()
        z_fixed_actual = available_z[np.argmin(np.abs(available_z - z_fixed))]
        df_z = df_multi[df_multi["z"] == z_fixed_actual]
        
        x = df_z["x"].values
        y = df_z["y"].values
        z = df_z["aging_factor"].values
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if len(x) >= 3:
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Aging Factor")
        ax.set_title(f"Aging Factor vs X-Y (z={z_fixed_actual:.2f}, {len(selected_events)} events)")
        fig.savefig(self.results_dir / filename.replace('.png', '_3d.png'), dpi=dpi, bbox_inches="tight")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(df_z["x"], df_z["y"], c=df_z["aging_factor"], cmap="viridis", s=50)
        plt.colorbar(scatter, label="Aging Factor")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"Aging Factor vs X-Y (z={z_fixed_actual:.2f}, {len(selected_events)} events)")
        plt.tight_layout()
        fig2.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        
        return fig2
    
    def _plot_aging_factor_z(
        self,
        dataset: pd.DataFrame,
        x_fixed: float = None,
        y_fixed: float = None,
        n_events: int = 1000,
        filename: str = "aging_factor_z.png",
        dpi: int = 150
    ):
        """Построение зависимости aging factor от Z для фиксированных X и Y."""
        event_counts = dataset.groupby("event")["cell_key"].nunique()
        events_with_multiple_cells = event_counts[event_counts > 1].index
        selected_events = np.random.choice(
            events_with_multiple_cells,
            size=min(n_events, len(events_with_multiple_cells)),
            replace=False
        )
        df_multi = dataset[dataset["event"].isin(selected_events)]
        
        if x_fixed is None or y_fixed is None:
            x_fixed = df_multi.groupby("x").size().idxmax()
            y_fixed = df_multi.groupby("y").size().idxmax()
        else:
            available_x = df_multi["x"].unique()
            available_y = df_multi["y"].unique()
            x_fixed = available_x[np.argmin(np.abs(available_x - x_fixed))]
            y_fixed = available_y[np.argmin(np.abs(available_y - y_fixed))]
        
        df_xy = df_multi[(df_multi["x"] == x_fixed) & (df_multi["y"] == y_fixed)]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df_xy["z"], df_xy["aging_factor"], s=50, alpha=0.6)
        ax.set_xlabel("Z")
        ax.set_ylabel("Aging Factor")
        ax.set_title(f"Aging Factor vs Z (x={x_fixed:.2f}, y={y_fixed:.2f}, {len(selected_events)} events)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        
        return fig
    
    def _plot_energy_histogram_cell(
        self,
        dataset: pd.DataFrame,
        cell_stats: pd.DataFrame,
        cell_id,
        filename: str = "energy_histogram_cell.png",
        dpi: int = 150
    ):
        """Построение гистограмм распределения энергии для выбранной ячейки."""
        df_cell = dataset[dataset['cell_key'] == cell_id]
        if df_cell.empty:
            raise ValueError(f"Ячейка с cell_id={cell_id} не найдена в данных")
        
        aging_factor = cell_stats.at[cell_id, 'aging']
        title = f"Histogram for CellID {cell_id}, Aging factor: [{aging_factor:.8f}]"
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.hist(df_cell['E_new'], bins=50, label='E (original)', histtype='step', linewidth=2)
        plt.hist(df_cell['E_old'], bins=50, label='E_aged', histtype='step', linewidth=2)
        plt.xlabel('$Energy\ [GeV]$', fontsize=12)
        plt.ylabel('$Frequency\ /\ GeV$', fontsize=12)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig1.savefig(self.results_dir / filename.replace('.png', '_full.png'), dpi=dpi, bbox_inches="tight")
        
        ran = [0, 2e3]
        fig2 = plt.figure(figsize=(10, 5))
        plt.hist(df_cell['E_new'], bins=50, range=ran, label='E (original)', histtype='step', linewidth=2)
        plt.hist(df_cell['E_old'], bins=50, range=ran, label='E_aged', histtype='step', linewidth=2)
        plt.xlabel('$Energy\ [GeV]$', fontsize=12)
        plt.ylabel('$Frequency\ /\ GeV$', fontsize=12)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig2.savefig(self.results_dir / filename, dpi=dpi, bbox_inches="tight")
        
        return fig2
