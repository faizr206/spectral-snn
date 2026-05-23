#!/usr/bin/env bash
# GPU 3: Multi-dataset — S-CIFAR10 and S-MNIST
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/root/.local/bin/uv
mkdir -p $PR/runs/logs

echo "=== S-CIFAR10, paper_real_shortlist, seeds 0-2 ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite paper_real_shortlist \
        --dataset scifar10 \
        --save-dir $PR/runs/real_scifar10/seed${SEED} \
        --seed $SEED --epochs 100 \
        --batch-size 64 --num-workers 4 \
        --device cuda --amp \
        --diagnostics-every 10 \
        --suite-parallelism 2
done

echo "=== S-MNIST, paper_real_shortlist, seeds 0-2 ==="
for SEED in 0 1 2; do
    $UV run python -m drf_experiment.cli \
        --suite paper_real_shortlist \
        --dataset smnist \
        --save-dir $PR/runs/real_smnist/seed${SEED} \
        --seed $SEED --epochs 50 \
        --batch-size 128 --num-workers 4 \
        --device cuda --amp \
        --suite-parallelism 3
done

echo "GPU3 DATASETS DONE"
