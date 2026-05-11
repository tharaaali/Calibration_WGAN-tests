#!/bin/bash
#SBATCH --job-name=wgan_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=0
#SBATCH --time=1-00:00:00
#SBATCH -A proj_1819
#SBATCH --output=/home/akarbyshev/Calibration_WGAN-tests/task-logs/wgan_%A_%a_out.txt
#SBATCH --error=/home/akarbyshev/Calibration_WGAN-tests/task-logs/wgan_%A_%a_err.txt
#SBATCH --partition=normal

set -euo pipefail

module purge

set +u
source /home/akarbyshev/.bashrc || true
set -u

CONDA_PY="/home/akarbyshev/miniconda3/envs/myenv/bin/python"

if [ -n "${1:-}" ]; then
    RUN_NAME="$1"
elif [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    RUN_NAME="run_${SLURM_ARRAY_JOB_ID}"
else
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

EXPERIMENT_NAME="${2:-ws_final}"

echo "Running on node: $(hostname)"
echo "Using python: $CONDA_PY"
echo "Run name: $RUN_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
$CONDA_PY -V

cd /home/akarbyshev/Calibration_WGAN-tests || exit 1
mkdir -p task-logs

CONFIGS_DIR="configs/generated/${EXPERIMENT_NAME}"
CONFIGS=("${CONFIGS_DIR}"/*.yaml)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with: sbatch --array=1-${#CONFIGS[@]} run_training.sh"
    exit 1
fi

idx=$((SLURM_ARRAY_TASK_ID - 1))
if (( idx < 0 || idx >= ${#CONFIGS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}. Number of configs: ${#CONFIGS[@]}"
    exit 1
fi

config="${CONFIGS[$idx]}"
exp_name=$(basename "$config" .yaml)
echo "Array task ${SLURM_ARRAY_TASK_ID} -> config: $exp_name"
echo "=== Running: $exp_name ==="
srun $CONDA_PY -u scripts/train_wgan.py --config "$config" --experiment "$EXPERIMENT_NAME" --config-name "$exp_name" --run-name "$RUN_NAME"

echo "All experiments completed!"
