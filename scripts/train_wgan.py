#!/usr/bin/env python3
"""Точка входа для запуска обучения WGAN."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from lib.train_loop import train_wgan
from lib.train_utils import setup_logging


def main():
    """Парсит аргументы и запускает обучение."""
    parser = argparse.ArgumentParser(description="Обучение WGAN для калибровки калориметра")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "configs" / "config.yaml"),
        help="Путь к файлу конфигурации",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Название эксперимента (для логов)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Название общего запуска для группировки результатов",
    )
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    experiment_name = args.experiment or datetime.now().strftime("wgan_%Y%m%d_%H%M%S")
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir = Path(config.get("results_dir", "results")).parent
    logs_dir = base_dir / config.get("wgan_params", {}).get("logs_dir", "logs") / run_name / experiment_name
    logger = setup_logging(logs_dir, experiment_name)
    
    logger.info("=" * 60)
    logger.info(f"WGAN Training - {experiment_name}")
    logger.info(f"Run name: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Конфиг: {args.config}")
    
    try:
        train_wgan(config, logger, experiment_name, run_name)
        logger.info("Обучение успешно завершено")
        return 0
    except Exception as error:
        logger.exception(f"Ошибка при обучении: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
