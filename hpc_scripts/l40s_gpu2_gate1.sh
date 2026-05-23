#!/usr/bin/env bash
# GPU 2: Gate1 verification — baseline + SRG at 200 epochs on SHD
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/root/.local/bin/uv
mkdir -p $PR/runs/logs

echo "=== Gate1: SHD 200ep, seeds 0-2 ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite gate1_verification \
        --dataset shd \
        --data-root $PR/data \
        --save-dir $PR/runs/gate1_200ep/seed${SEED} \
        --seed $SEED --epochs 200 \
        --batch-size 128 --num-workers 4 \
        --device cuda --amp \
        --diagnostics-every 10 \
        --suite-parallelism 1
done

echo "GPU2 GATE1 DONE"
