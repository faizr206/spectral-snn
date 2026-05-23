#!/usr/bin/env bash
# GPU 0: H3 isolation suite — the most critical experiment
# Does spectral scoring matter beyond top-k sparsity?
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/root/.local/bin/uv
mkdir -p $PR/runs/logs

echo "=== H3 isolation: SHD, 3 seeds ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite h3_isolation \
        --dataset shd \
        --data-root $PR/data \
        --save-dir $PR/runs/h3_shd/seed${SEED} \
        --seed $SEED --epochs 100 \
        --batch-size 128 --num-workers 4 \
        --device cuda --amp \
        --diagnostics-every 10 \
        --suite-parallelism 1
done

echo "=== H3 isolation: sine_frequency, 3 seeds ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite h3_isolation \
        --dataset sine_frequency \
        --save-dir $PR/runs/h3_sine/seed${SEED} \
        --seed $SEED --epochs 50 \
        --batch-size 128 --num-workers 4 \
        --device cuda --amp \
        --suite-parallelism 3
done

echo "GPU0 H3 DONE"
