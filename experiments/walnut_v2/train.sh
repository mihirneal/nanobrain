#!/usr/bin/env bash
# experiments/walnut_v2/train.sh [ACCOUNT] [NAME]
#   Walnut v2 release training: src/nanobrain/config/walnut_v2.yaml on 8 GPUs (DDP, may span nodes), run from
#   this git checkout itself (no source copy), so the log and summary record its commit. Refuses a dirty tree.
set -euo pipefail
ACCOUNT=${1:-max}; NAME=${2:-walnut_v2}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-/data/mihir.neal/nanobrain_vitl_p4/runs}
PY=${PY:-/admin/home/mihir.neal/nanobrain/.venv/bin/python}
STATUS=$(git -C "$REPO" status --porcelain)
if [[ -n $STATUS ]]; then echo "refusing: $REPO is not a clean checkout"; echo "$STATUS"; exit 1; fi
SHA=$(git -C "$REPO" rev-parse HEAD)
echo "training $NAME from $REPO at $SHA"
mkdir -p "$OUT/slurms"
sbatch --parsable --job-name="$NAME" --partition=n --account="$ACCOUNT" --qos=normal \
  --nodes=1-4 --ntasks=8 --gpus-per-task=1 --cpus-per-task=16 --mem-per-cpu=8G --time=20:00:00 \
  --requeue --open-mode=append --output="$OUT/slurms/${NAME}_%j.out" --wrap "
cd $REPO
export MASTER_ADDR=\$(scontrol show hostnames \"\$SLURM_JOB_NODELIST\" | head -1)
export MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
export NCCL_SOCKET_IFNAME=^docker,lo,br-,veth NCCL_NVLS_ENABLE=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_\$SLURM_JOB_ID WANDB_DIR=$OUT/$NAME BRAINMARKS_SMRI_DATA=/data/brainmarks-smri PYTHONPATH=$REPO/src
srun --kill-on-bad-exit=1 --gpu-bind=none bash -c '
  export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS
  unset CUDA_VISIBLE_DEVICES
  ngpu=\$(nvidia-smi -L | wc -l)
  if (( ngpu > SLURM_LOCALID )); then export LOCAL_RANK=\$SLURM_LOCALID; else export LOCAL_RANK=0; fi
  exec $PY -m nanobrain.train --config src/nanobrain/config/walnut_v2.yaml \
    --overrides output_dir=$OUT name=$NAME distributed=true wandb=true wandb_project=nanobrain-vitl-p4'
"
