#!/usr/bin/env bash
# Wait until SHD data is ready AND a GPU is free, then launch H3 isolation.
# Keeps at least one GPU empty at all times (never uses GPU 1 = VLLM).
# Run this in the background on the L40S login node.
set -euo pipefail
PR=/opt/dlami/nvme/noel/spectral_snn
UV=/home/ubuntu/.local/bin/uv
LOG=$PR/runs/logs

mkdir -p $LOG

echo "[waitlaunch] Waiting for SHD frames and a free GPU..."

while true; do
    # Check SHD data readiness (need >100 npy files = at least some classes done)
    NP=$(find $PR/data/frames_number_250_split_by_number/train -name '*.npy' 2>/dev/null | wc -l)

    # Find free GPUs (exclude GPU 1 = VLLM, look for < 1000 MiB usage)
    FREE_GPUS=()
    while IFS=',' read -r idx used; do
        idx=$(echo $idx | tr -d ' ')
        used=$(echo $used | tr -d ' MiB')
        if [[ "$idx" != "1" && "$used" -lt 1000 ]]; then
            FREE_GPUS+=($idx)
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

    echo "[waitlaunch] SHD frames: $NP, Free GPUs: ${FREE_GPUS[*]:-none}"

    # Need SHD ready AND >= 2 free GPUs (so we can use 1 and keep 1 empty)
    if [[ "$NP" -gt 100 && "${#FREE_GPUS[@]}" -ge 2 ]]; then
        GPU=${FREE_GPUS[0]}
        echo "[waitlaunch] Launching H3 isolation on GPU $GPU. Keeping GPU ${FREE_GPUS[1]} empty."
        export CUDA_VISIBLE_DEVICES=$GPU
        export HF_HOME=/opt/dlami/nvme/hf_cache
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PYTHONUNBUFFERED=1
        for SEED in 0 1 2; do
            $UV run python -m drf_experiment.cli \
                --suite h3_isolation \
                --dataset shd \
                --data-root $PR/data \
                --save-dir $PR/runs/h3_shd_v2/seed${SEED} \
                --seed $SEED --epochs 100 \
                --batch-size 128 --num-workers 4 \
                --device cuda --amp \
                --diagnostics-every 10 \
                --suite-parallelism 1 \
                2>&1 | tee -a $LOG/gpu${GPU}_h3_seed${SEED}.log
        done
        echo "[waitlaunch] H3 isolation complete."
        break
    fi

    sleep 60
done
