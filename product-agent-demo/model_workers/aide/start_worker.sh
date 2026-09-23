#!/usr/bin/env bash
set -eo pipefail

AIDE_CONDA_SH="${AIDE_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
AIDE_CONDA_ENV="${AIDE_CONDA_ENV:-aide}"
source "$AIDE_CONDA_SH"
conda activate "$AIDE_CONDA_ENV"
set -u
export AIDE_REPO="${AIDE_REPO:-${HOME}/agent3-models/AIDE-main}"
export AIDE_CHECKPOINT="${AIDE_CHECKPOINT:-${HOME}/agent3-models/checkpoints/aide/GenImage_train.pth}"
export AIDE_DEVICE="${AIDE_DEVICE:-cuda}"
export AIDE_MODEL_VERSION="${AIDE_MODEL_VERSION:-aide-genimage-v1}"
export AIDE_PORT="${AIDE_PORT:-8101}"
cd "$(dirname "$0")"
exec python -m uvicorn app:app --host 0.0.0.0 --port "$AIDE_PORT"
