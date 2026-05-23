#!/usr/bin/env bash
# H11: Wall-clock timing test — baseline_drf vs gate_TopK2_SRG_fast on S-CIFAR10 T=1024
# Run 3 epochs each to get stable epoch_time_sec. No accuracy convergence needed.
# If gate_TopK2_SRG_fast is 2-4x faster, SRG provides real GPU speedup on standard hardware.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/opt/dlami/nvme/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/home/ubuntu/.local/bin/uv

echo "=== H11 timing: baseline_drf on S-CIFAR10 T=1024 (3 epochs) ==="
$UV run python -m drf_experiment.cli \
    --variant baseline_drf \
    --dataset scifar10 \
    --save-dir $PR/runs/h11_timing \
    --seed 0 --epochs 3 \
    --batch-size 128 --num-workers 4 \
    --device cuda --amp \
    --eval-every 5 \
    --diagnostics-every 10

echo "=== H11 timing: gate_TopK2_SRG_fast on S-CIFAR10 T=1024 (3 epochs) ==="
$UV run python -m drf_experiment.cli \
    --variant gate_TopK2_SRG_fast \
    --dataset scifar10 \
    --save-dir $PR/runs/h11_timing \
    --seed 0 --epochs 3 \
    --batch-size 128 --num-workers 4 \
    --device cuda --amp \
    --eval-every 5 \
    --diagnostics-every 10

echo "=== H11 timing results ==="
$UV run python -c "
import json, glob
for f in sorted(glob.glob('$PR/runs/h11_timing/*/metrics.json')):
    d = json.load(open(f))
    h = d.get('history', [])
    name = d.get('config',{}).get('name','?')
    if h:
        times = [ep.get('train',{}).get('epoch_time_sec',0) for ep in h]
        avg = sum(times)/len(times)
        print(f'{name}: avg {avg:.1f}s/epoch = {avg/60:.1f} min/epoch over {len(times)} epochs')
"
echo "GPU3 H11 DONE"
