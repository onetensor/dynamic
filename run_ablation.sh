#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TORCHRUN_ARGS="${TORCHRUN_ARGS:-}"

RUN_BASE="${RUN_BASE:-ablations/$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_ID="${BASE_RUN_ID:-${RUN_BASE}/softmax}"
DROP_RUN_ID="${DROP_RUN_ID:-${RUN_BASE}/dropsoftmax}"

export NPROC_PER_NODE
export TORCHRUN_ARGS

echo "Running baseline: ${BASE_RUN_ID}"
HP_RUN_ID="${BASE_RUN_ID}" TRAIN_SCRIPT="train_gpt.py" bash run.sh

echo "Running DropSoftmax: ${DROP_RUN_ID}"
HP_RUN_ID="${DROP_RUN_ID}" TRAIN_SCRIPT="train_gpt_drop.py" bash run.sh
