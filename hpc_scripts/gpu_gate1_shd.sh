#!/usr/bin/env bash
# Gate 1 — Verify baseline energy claim on SHD
# Runs: baseline_drf, gate_D1 (MLP), gate_SRG — 3 seeds each, pytorch backend
# Expected: ~6-10h on GPU. Results in runs/gate1_shd/
#SBATCH --job-name=snn_gate1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=runs/slurm_logs/%j.out
#SBATCH --error=runs/slurm_logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noel.thomas@mbzuai.ac.ae
set -euo pipefail
[ -n "${SLURM_SUBMIT_DIR:-}" ] && PR="${SLURM_SUBMIT_DIR}" || PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PR}/runs/slurm_logs" "${PR}/runs/gate1_shd" "${PR}/data"
command -v uv &>/dev/null && RUN="uv run python" || { [ -d "${PR}/.venv" ] && source "${PR}/.venv/bin/activate"; RUN="python3 -u"; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "PyTorch CUDA: $(${RUN} -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))')"
cd "${PR}"

# Download SHD data if not present (SpikingJelly auto-downloads)
echo "=== Checking SHD data ==="
${RUN} -c "
from drf_experiment.datasets import build_dataloaders
from drf_experiment.config import DatasetConfig
cfg = DatasetConfig(name='shd', root='./data', batch_size=64, num_workers=0)
try:
    build_dataloaders(cfg)
    print('SHD data OK')
except Exception as e:
    print(f'SHD download error: {e}')
"

echo "=== Gate 1: baseline_drf on SHD (3 seeds) ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --variant baseline_drf --dataset shd \
        --data-root ./data \
        --save-dir ./runs/gate1_shd \
        --seed ${SEED} --epochs 50 \
        --batch-size 64 --num-workers 4 \
        --device cuda --amp
done

echo "=== Gate 1: gate_D1 (MLP gate) on SHD (3 seeds) ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --variant gate_D1 --dataset shd \
        --data-root ./data \
        --save-dir ./runs/gate1_shd \
        --seed ${SEED} --epochs 50 \
        --batch-size 64 --num-workers 4 \
        --device cuda --amp
done

echo "=== Gate 1: gate_SRG (spectral resonance) on SHD (3 seeds) ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --variant gate_SRG --dataset shd \
        --data-root ./data \
        --save-dir ./runs/gate1_shd \
        --seed ${SEED} --epochs 50 \
        --batch-size 64 --num-workers 4 \
        --device cuda --amp
done

echo "=== Gate 1: gate_TopK2_SRG on SHD (3 seeds) ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --variant gate_TopK2_SRG --dataset shd \
        --data-root ./data \
        --save-dir ./runs/gate1_shd \
        --seed ${SEED} --epochs 50 \
        --batch-size 64 --num-workers 4 \
        --device cuda --amp
done

echo "=== Generating summary ==="
${RUN} -m drf_experiment.cli --plot-root ./runs/gate1_shd --plot-output ./runs/gate1_shd/plots
echo "Gate 1 complete. Results in runs/gate1_shd/"
