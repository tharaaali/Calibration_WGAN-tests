#!/bin/bash
#SBATCH --job-name=wgan_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/akarbyshev/Calibration_WGAN-tests/task-logs/wgan_%j_out.txt
#SBATCH --error=/home/akarbyshev/Calibration_WGAN-tests/task-logs/wgan_%j_err.txt
#SBATCH --partition=normal

set -euo pipefail

module purge

set +u
source /home/akarbyshev/.bashrc || true
set -u

CONDA_PY="/home/akarbyshev/miniconda3/envs/myenv/bin/python"
RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"

echo "Running on node: $(hostname)"
echo "Using python: $CONDA_PY"
echo "Run name: $RUN_NAME"
$CONDA_PY -V

cd /home/akarbyshev/Calibration_WGAN-tests || exit 1
mkdir -p task-logs

for config in configs/generated/*.yaml; do
    exp_name=$(basename "$config" .yaml)
    echo "=== Running: $exp_name ==="
    srun $CONDA_PY -u scripts/train_wgan.py --config "$config" --experiment "$exp_name" --run-name "$RUN_NAME"
done

echo "All experiments completed!"
