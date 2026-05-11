# Calibration WGAN

WGAN для калибровки калориметра: оценка коэффициентов старения (aging factors) ячеек по событиям до/после деградации детектора.

## Структура

```
.
├── configs/
│   ├── config.yaml              # базовый конфиг (модель, данные, WGAN, графики)
│   └── generated/               # наборы конфигов под отдельные эксперименты
│       └── <experiment>/config_*.yaml
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
│   ├── models.ipynb             # эксперименты и анализ бейзлайн моделей
│   ├── compare_runs_metrics.ipynb  # сравнение метрик между запусками
│   └── generate_configs.ipynb   # генерация наборов конфигов для серий экспериментов
├── models.ipynb                 # ноутбук верхнего уровня
├── run_training.sh              # SLURM-скрипт (sbatch array) для запуска серии
├── results/                     # выгрузки запусков: предсказания, кривые, гистограммы
├── data/                        # входные данные (gitignored)
├── logs/, models/, task-logs/   # логи и чекпоинты (gitignored)
└── .cometml-runs/               # локальные кэши Comet ML (gitignored)
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

## Конфиг

Ключевые секции `configs/config.yaml`:

- `data_path`, `results_dir` — пути к данным и выгрузкам.
- `model` — `ws` или `wgan`.
- `aging` — параметры генератора aging-факторов (функция, диапазон, шум, сиды, разбиение).
- `wgan_params` — гиперпараметры WGAN: `batch_size`, `epochs`, `lr_g`/`lr_d`, `n_critic`, `gp_lambda`, scheduler, energy mask/cuts, фильтр по геометрии.
- `plots` — параметры графиков (бины, квантили, размеры, имена файлов).

Серии конфигов под отдельные эксперименты лежат в `configs/generated/<experiment>/` и собираются ноутбуком `notebooks/generate_configs.ipynb`.

## Результаты

Каждый запуск сохраняет в `results/<run_name>/<experiment>/<config>/`:

- `wgan_predictions.csv` — предсказания на тесте,
- `wgan_training_history.csv` — история loss/метрик по эпохам,
- `wgan_training_curves.png`, `wgan_true_vs_predicted.png`, `wgan_error_histogram.png` — графики.

Сравнение метрик между запусками — `notebooks/compare_runs_metrics.ipynb`.