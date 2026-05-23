#!/usr/bin/env bash
# S-CIFAR10 fast run: GPU 3 only, 30 epochs, batch 128, eval every 5
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/home/ubuntu/.local/bin/uv

echo "=== S-CIFAR10: 30ep, batch=128, parallelism=1 ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite paper_real_shortlist \
        --dataset scifar10 \
        --save-dir $PR/runs/real_scifar10_v2/seed${SEED} \
        --seed $SEED --epochs 30 \
        --batch-size 128 --num-workers 4 \
        --device cuda --amp \
        --eval-every 5 \
        --diagnostics-every 30 \
        --suite-parallelism 1
done
echo "DONE"
