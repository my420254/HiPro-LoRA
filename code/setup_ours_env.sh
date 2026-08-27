#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${1:-hipro-lora}"
CONDA_SH="${2:-${CONDA_PREFIX}/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SH}" ]]; then
  CONDA_SH="${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" 2>/dev/null || true
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  echo "Usage: $0 [env_name] [path/to/conda.sh]" >&2
  exit 1
fi

source "${CONDA_SH}"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] create env ${ENV_NAME}" >&2
  conda create -y -n "${ENV_NAME}" python=3.10
else
  echo "[setup] env ${ENV_NAME} already exists" >&2
fi

echo "[setup] upgrade pip" >&2
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "[setup] install torch/cu128" >&2
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.7.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "[setup] install project deps" >&2
conda run -n "${ENV_NAME}" python -m pip install \
  transformers==4.53.3 \
  datasets==4.4.1 \
  peft==0.16.0 \
  scikit-learn==1.7.2 \
  pandas==2.3.3 \
  numpy==2.2.5 \
  matplotlib==3.10.9 \
  seaborn==0.12.2 \
  accelerate==1.13.0 \
  tqdm==4.67.1 \
  sentencepiece

PREFETCH_SCRIPT="${PROJECT_DIR}/prefetch_ours_resources.py"

echo "[setup] prefetch encoder models and datasets via mirror" >&2
if [[ -f "${PREFETCH_SCRIPT}" ]]; then
  HF_ENDPOINT="https://hf-mirror.com" \
  TOKENIZERS_PARALLELISM="false" \
  conda run -n "${ENV_NAME}" python "${PREFETCH_SCRIPT}"
else
  echo "[setup] prefetch script not found, skipping: ${PREFETCH_SCRIPT}" >&2
fi

cat <<EOF

[setup] done

activate:
  source ${CONDA_SH}
  conda activate ${ENV_NAME}

runtime env:
  export HF_ENDPOINT=https://hf-mirror.com
  export TOKENIZERS_PARALLELISM=false
  export NVIDIA_TF32_OVERRIDE=0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8

EOF
