#!/usr/bin/env bash
set -euo pipefail

# Install FlashInfer SM89 build/runtime dependencies into a conda environment.
#
# Usage:
#   bash scripts/build/sm89_install_deps.sh          # use the active conda env
#   ENV_NAME=flashinfer-sm89 bash scripts/build/sm89_install_deps.sh
#
# Optional env vars:
#   ENV_NAME        conda env to create/reuse (default: active env)
#   PYTHON_VERSION  python version when creating an env (default: 3.12)
#   TORCH_VERSION   torch pin, e.g. 2.13.0+cu130 (default: latest cu130 wheel)
#   TORCH_INDEX_URL PyTorch cu130 index (default: https://download.pytorch.org/whl/cu130)
#   PYPI_INDEX_URL  PyPI index mirror, e.g. https://pypi.tuna.tsinghua.edu.cn/simple
#   BUILD_NIXL_EP / BUILD_NCCL_EP   passed through to the build backend
#                   (default 0 for SM89 sparse MLA; see sm89_env.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Locate the python interpreter (active conda env by default).
if [[ -n "${ENV_NAME:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "error: ENV_NAME=${ENV_NAME} requested but conda is not on PATH" >&2
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  CONDA_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
  if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION:-3.12})..."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION:-3.12}"
  fi
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "error: no conda env active; activate one or set ENV_NAME=..." >&2
    exit 1
  fi
  PYTHON="${CONDA_PREFIX}/bin/python"
fi

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

echo "==> Python: ${PYTHON} ($("${PYTHON}" --version 2>&1))"
"${PYTHON}" -m pip install -U pip

echo "==> Installing build backend requirements (needed for --no-build-isolation)"
"${PYTHON}" -m pip install -U build ninja setuptools packaging \
  "apache-tvm-ffi>=0.1.6,!=0.1.8,!=0.1.8.post0,<0.2"

echo "==> Installing torch from ${TORCH_INDEX_URL}"
"${PYTHON}" -m pip install "torch${TORCH_VERSION:+==${TORCH_VERSION}}" \
  --index-url "${TORCH_INDEX_URL}"

echo "==> Installing remaining runtime requirements (torch line stripped)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
sed -E '/^(torch)([[:space:]]*[<>=]|$)/d' \
  "${REPO_ROOT}/requirements.txt" > "${TMP}/requirements.txt"
"${PYTHON}" -m pip install ${PYPI_INDEX_URL:+-i "${PYPI_INDEX_URL}"} \
  -r "${TMP}/requirements.txt"

echo "==> Done. Next steps:"
echo "    bash scripts/build/sm89_prepare_deps.sh"
echo "    bash scripts/build/sm89_build_wheel.sh --editable   # dev install"
