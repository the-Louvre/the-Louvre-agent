#!/usr/bin/env bash
# Start the IML-ViT Worker on the Ubuntu teaching server.
set -eo pipefail

IML_VIT_CONDA_SH="${IML_VIT_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
IML_VIT_CONDA_ENV="${IML_VIT_CONDA_ENV:-imlvit}"
source "$IML_VIT_CONDA_SH"
conda activate "$IML_VIT_CONDA_ENV"
set -u

export IML_VIT_REPO="${IML_VIT_REPO:-${HOME}/agent3-models/IML-ViT-main}"
export IML_VIT_CHECKPOINT="${IML_VIT_CHECKPOINT:-${HOME}/agent3-models/checkpoints/iml_vit/iml-vit_checkpoint.pth}"
export IML_VIT_DEVICE="${IML_VIT_DEVICE:-cuda}"
export IML_VIT_MODEL_VERSION="${IML_VIT_MODEL_VERSION:-iml-vit-v1}"
export IML_VIT_PORT="${IML_VIT_PORT:-8102}"

cd "$(dirname "$0")"
exec python -m uvicorn app:app --host 0.0.0.0 --port "$IML_VIT_PORT"
