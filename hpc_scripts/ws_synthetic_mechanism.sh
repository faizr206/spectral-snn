#!/usr/bin/env bash
# Synthetic mechanism — paper_synthetic_mechanism suite, pytorch D-RF backend
# Sanity check + SRG mechanism figure on sine_frequency and chirp datasets
# Expected: ~2-3h on ws-ia. Results in runs/synthetic_mechanism/
#SBATCH --job-name=snn_synth
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=runs/slurm_logs/%j.out
#SBATCH --error=runs/slurm_logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noel.thomas@mbzuai.ac.ae
set -euo pipefail
[ -n "${SLURM_SUBMIT_DIR:-}" ] && PR="${SLURM_SUBMIT_DIR}" || PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PR}/runs/slurm_logs" "${PR}/runs/synthetic_mechanism"
command -v uv &>/dev/null && RUN="uv run python" || { [ -d "${PR}/.venv" ] && source "${PR}/.venv/bin/activate"; RUN="python3 -u"; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
cd "${PR}"

echo "=== Spectral diagnostic: sine_frequency ==="
${RUN} -m drf_experiment.cli \
    --dataset-diagnostic --dataset sine_frequency \
    --diagnostic-output ./runs/synthetic_mechanism/sine_freq_diagnostic.json

echo "=== Spectral diagnostic: chirp ==="
${RUN} -m drf_experiment.cli \
    --dataset-diagnostic --dataset chirp \
    --diagnostic-output ./runs/synthetic_mechanism/chirp_diagnostic.json

echo "=== paper_synthetic_mechanism suite: sine_frequency, 3 seeds, pytorch ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --suite paper_synthetic_mechanism \
        --dataset sine_frequency \
        --implementation pytorch \
        --save-dir ./runs/synthetic_mechanism/seed${SEED} \
        --seed ${SEED} --epochs 30 \
        --batch-size 128 --num-workers 4 \
        --device cuda --suite-parallelism 4
done

echo "=== paper_synthetic_mechanism suite: chirp, 3 seeds, pytorch ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --suite paper_synthetic_mechanism \
        --dataset chirp \
        --implementation pytorch \
        --save-dir ./runs/synthetic_mechanism/chirp_seed${SEED} \
        --seed ${SEED} --epochs 30 \
        --batch-size 128 --num-workers 4 \
        --device cuda --suite-parallelism 4
done

echo "=== Generating summary ==="
${RUN} -m drf_experiment.cli --plot-root ./runs/synthetic_mechanism --plot-output ./runs/synthetic_mechanism/plots
echo "Synthetic mechanism complete. Results in runs/synthetic_mechanism/"
