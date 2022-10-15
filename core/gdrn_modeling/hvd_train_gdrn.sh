#!/usr/bin/env bash
set -x
this_dir=$(dirname "$0")
# commonly used opts:

# MODEL.WEIGHTS: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
# CUDA_LAUNCH_BLOCKING=1
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
# NOTE: the custom evaluator after hvd training is buggy currently
# either run evaluation with single gpu or use BOP evaluator
# autotune hvd parameters for speed
#CUDA_VISIBLE_DEVICES=$2 horovodrun -np $NGPU --autotune \
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$2 horovodrun -np $NGPU --autotune -H localhost:$NGPU \
    python $this_dir/main_gdrn.py \
    --config-file $CFG --num-gpus $NGPU  --launcher hvd ${@:3}
