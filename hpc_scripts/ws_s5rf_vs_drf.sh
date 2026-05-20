#!/usr/bin/env bash
# S5-RF vs D-RF comparison — paper_synthetic_mechanism suite, jax-ssm backend
# Establishes the SSM baseline for "what SSMs cannot do" framing
# Expected: ~3-4h on ws-ia. Results in runs/s5rf_comparison/
#SBATCH --job-name=snn_s5rf
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm_logs/%j.out
#SBATCH --error=runs/slurm_logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noel.thomas@mbzuai.ac.ae
set -euo pipefail
[ -n "${SLURM_SUBMIT_DIR:-}" ] && PR="${SLURM_SUBMIT_DIR}" || PR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PR}/runs/slurm_logs" "${PR}/runs/s5rf_comparison"
command -v uv &>/dev/null && RUN="uv run python" || { [ -d "${PR}/.venv" ] && source "${PR}/.venv/bin/activate"; RUN="python3 -u"; }
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
cd "${PR}"

# Install JAX stack if not present
echo "=== Checking JAX stack ==="
${RUN} -c "import jax; import equinox; import optax; print('JAX', jax.__version__, 'OK')" 2>/dev/null || {
    echo "Installing JAX stack..."
    uv pip install "jax[cuda12]>=0.4.30" "equinox>=0.11" "optax>=0.2" -q
}
${RUN} -c "import jax; print('JAX devices:', jax.devices())"

echo "=== paper_synthetic_mechanism: sine_frequency, jax-ssm backend, 3 seeds ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --suite paper_synthetic_mechanism \
        --dataset sine_frequency \
        --implementation jax-ssm \
        --save-dir ./runs/s5rf_comparison/sine_seed${SEED} \
        --seed ${SEED} --epochs 30 \
        --batch-size 128
done

echo "=== paper_synthetic_mechanism: chirp, jax-ssm backend, 3 seeds ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --suite paper_synthetic_mechanism \
        --dataset chirp \
        --implementation jax-ssm \
        --save-dir ./runs/s5rf_comparison/chirp_seed${SEED} \
        --seed ${SEED} --epochs 30 \
        --batch-size 128
done

echo "=== S5-RF on SHD (baseline only, 3 seeds) ==="
for SEED in 0 1 2; do
    ${RUN} -m drf_experiment.cli \
        --variant baseline_drf --dataset shd \
        --implementation jax-ssm \
        --data-root ./data \
        --save-dir ./runs/s5rf_comparison/shd_seed${SEED} \
        --seed ${SEED} --epochs 50 \
        --batch-size 64
done

echo "S5-RF comparison complete. Results in runs/s5rf_comparison/"
