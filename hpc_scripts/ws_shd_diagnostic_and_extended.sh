#!/usr/bin/env bash
# SHD spectral diagnostic + extended training (200 ep, lr=0.004, SHD-tuned)
# Gate 1 real verification — matches D-RF paper hyperparameters
# Expected: ~5 min diagnostic + ~6h training (200 ep × 36s = 2h per run, 3 variants)
#SBATCH --job-name=snn_shd_ext
#SBATCH --partition=ws-ia
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
mkdir -p "${PR}/runs/slurm_logs" "${PR}/runs/shd_extended" "${PR}/runs/shd_diagnostic"
command -v uv &>/dev/null && RUN="uv run python" || { [ -d "${PR}/.venv" ] && source "${PR}/.venv/bin/activate"; RUN="python3 -u"; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
cd "${PR}"

# Step 1: SHD stationarity diagnostic (minutes, validates STFT angle)
echo "=== SHD spectral diagnostic ==="
${RUN} -m drf_experiment.cli \
    --dataset-diagnostic --dataset shd \
    --data-root ./data \
    --max-batches 16 --chunk-size 25 --hop-size 25 \
    --diagnostic-output ./runs/shd_diagnostic/shd_diagnostic.json
echo "SHD diagnostic done."

# Step 2: Extended training — 200 epochs, lr=0.004 (SHD-tuned)
# baseline_drf: 1 seed (most critical for Gate 1 energy verification)
echo "=== Extended: baseline_drf, SHD, 200 ep, lr=0.004, seed=0 ==="
${RUN} -m drf_experiment.cli \
    --variant baseline_drf --dataset shd \
    --data-root ./data \
    --save-dir ./runs/shd_extended \
    --seed 0 --epochs 200 \
    --batch-size 64 --num-workers 4 \
    --device cuda --amp

# gate_SRG: 1 seed (Gate 1 SRG comparison)
echo "=== Extended: gate_SRG, SHD, 200 ep, lr=0.004, seed=0 ==="
${RUN} -m drf_experiment.cli \
    --variant gate_SRG --dataset shd \
    --data-root ./data \
    --save-dir ./runs/shd_extended \
    --seed 0 --epochs 200 \
    --batch-size 64 --num-workers 4 \
    --device cuda --amp

echo "Extended SHD runs complete. Results in runs/shd_extended/"
