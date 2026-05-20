# Calibration WGAN

WGAN для калибровки калориметра: оценка коэффициентов старения (aging factors) ячеек по событиям до/после деградации детектора.

## Структура

```
.
├── configs/
│   └── config.yaml              # базовый конфиг (модель, данные, WGAN, графики)
├── scripts/
│   ├── train_wgan.py            # точка входа для обучения WGAN
│   ├── plot_graphics.py         # генерация графиков по данным
│   ├── utils.py                 # AgingFactorGenerator и общие утилиты
│   └── lib/
│       ├── dataset.py           # CaloEventDataset, разбиение данных
│       ├── models.py            # Generator / Discriminator
│       ├── train_loop.py        # цикл обучения WGAN (+GP, scheduler, etc)
│       ├── train_utils.py       # логирование, сохранение чекпоинтов
│       └── plots.py             # графики тренировок (loss, true vs pred и т.п.)
├── notebooks/
│   ├── baseline_models.ipynb        # базовые методы оценки коэффициентов (mean, median, Wasserstein)
│   ├── compare_runs_metrics.ipynb   # сравнение метрик между запусками WGAN
│   └── generate_configs.ipynb       # генерация наборов конфигов для серий экспериментов
└── run_training.sh              # SLURM-скрипт (sbatch array) для запуска серии
```

## Запуск

Локально:

```bash
python scripts/train_wgan.py \
    --config configs/generated/<experiment>/config_<name>.yaml \
    --experiment <experiment> \
    --config-name config_<name> \
    --run-name run_$(date +%Y%m%d_%H%M%S)
```

На кластере через SLURM array (по одной задаче на каждый конфиг в `configs/generated/<experiment>/`):

```bash
sbatch --array=1-N run_training.sh <run_name> <experiment>
```

Конфиги для серий генерируются ноутбуком `notebooks/generate_configs.ipynb`.

## Конфиг

Ключевые секции `configs/config.yaml`:

- `data_path`, `results_dir` — пути к данным и выгрузкам.
- `model` — `ws` или `wgan`.
- `aging` — параметры генератора aging-факторов (функция, диапазон, шум, сиды, разбиение).
- `wgan_params` — гиперпараметры WGAN: `batch_size`, `epochs`, `lr_g`/`lr_d`, `n_critic`, `gp_lambda`, scheduler, energy mask/cuts, фильтр по геометрии.
- `plots` — параметры графиков (бины, квантили, размеры, имена файлов).

## Результаты

Каждый запуск сохраняет в `results/<run_name>/<experiment>/<config>/`:

- `wgan_predictions.csv` — предсказания на тесте,
- `wgan_training_history.csv` — история loss/метрик по эпохам,
- `wgan_training_curves.png`, `wgan_true_vs_predicted.png`, `wgan_error_histogram.png` — графики.

Сравнение метрик между запусками — `notebooks/compare_runs_metrics.ipynb`; базовые методы оценки коэффициентов (среднее, медиана, прямая минимизация расстояния Вассерштейна) — `notebooks/baseline_models.ipynb`.
