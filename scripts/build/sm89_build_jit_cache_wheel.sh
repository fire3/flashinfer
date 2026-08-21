#!/usr/bin/env bash
set -euo pipefail

# Build a FlashInfer JIT-cache wheel with all kernels precompiled for SM89.
#
# Produces a platform wheel (cp39-abi3) containing prebuilt .so modules;
# install it alongside flashinfer-python so users do not need nvcc/ninja at
# runtime. The marker matches the AOT wheel:
#   flashinfer_jit_cache-0.6.16.post3+sm89-...
#
# Usage:
#   bash scripts/build/sm89_build_jit_cache_wheel.sh
#
# Optional env vars: OUTPUT_DIR, SM89_WHEEL_MARKER, MAX_JOBS,
#                    FLASHINFER_NVCC_THREADS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKG_DIR="${REPO_ROOT}/flashinfer-jit-cache"
cd "${REPO_ROOT}"

source scripts/build/sm89_env.sh

if [[ "${FLASHINFER_CUDA_ARCH_LIST}" != "8.9" ]]; then
  echo "error: SM89 JIT-cache wheel requires FLASHINFER_CUDA_ARCH_LIST=8.9 (got ${FLASHINFER_CUDA_ARCH_LIST})" >&2
  exit 1
fi

# AOT compilation needs the CUTLASS/spdlog/CCCL headers at pinned commits.
bash scripts/build/sm89_prepare_deps.sh

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
WHEEL_MARKER="${SM89_WHEEL_MARKER:-sm89}"

echo "=========================================="
echo "Building FlashInfer SM89 JIT-cache wheel"
echo "=========================================="
echo "Repository: ${REPO_ROOT}"
echo "Python: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "CUDA arch list: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Wheel marker: ${WHEEL_MARKER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "MAX_JOBS: ${MAX_JOBS:-unset}"
echo "FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS:-unset}"
echo

# nvcc is required for AOT compilation (sm89_env.sh already tried to discover
# it; fail loudly here if still missing).
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found on PATH; install the CUDA toolkit or nvidia-cuda-nvcc" >&2
  exit 1
fi
echo "Using nvcc: $(command -v nvcc)"
"$(command -v nvcc)" --version | tail -n 3
echo

# The pip nvidia-cuda-nvcc wheel has no lib64/libcudart.so or libcuda.so, so
# the AOT link step cannot find -lcudart/-lcuda. Point them at the runtime
# wheel's libcudart.so.13 and the system driver stub (libcuda is only needed
# at link time; the loaded .so depends on libcudart.so.13 at runtime).
cuda_home=$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)
if [[ -f "${cuda_home}/lib/libcudart.so.13" && ! -e "${cuda_home}/lib64/libcudart.so" ]]; then
  mkdir -p "${cuda_home}/lib64"
  ln -s ../lib/libcudart.so.13 "${cuda_home}/lib64/libcudart.so"
  echo "linked ${cuda_home}/lib64/libcudart.so -> ../lib/libcudart.so.13"
fi
if [[ -f "/usr/local/cuda/lib64/stubs/libcuda.so" && ! -e "${cuda_home}/lib64/stubs/libcuda.so" ]]; then
  mkdir -p "${cuda_home}/lib64/stubs"
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so "${cuda_home}/lib64/stubs/libcuda.so"
  echo "linked ${cuda_home}/lib64/stubs/libcuda.so -> /usr/local/cuda/lib64/stubs/libcuda.so"
fi

python3 -m pip install --upgrade build

cd "${PKG_DIR}"

# Clean generated artifacts from previous AOT builds (stale .so modules would
# otherwise be swept into the wheel). *.so is gitignored, so this only removes
# build outputs.
rm -rf build dist *.egg-info flashinfer_jit_cache/jit_cache

# --skip-dependency-check: all build deps are installed by sm89_install_deps.sh;
# the frontend's resolver otherwise rejects torch's transitive nvjitlink pin.
python3 -m build --wheel --no-isolation --skip-dependency-check

wheel_path=$(find "${PKG_DIR}/dist" -maxdepth 1 -name "*.whl" | head -n 1)
if [[ -z "${wheel_path}" ]]; then
  echo "error: wheel build finished but no wheel was produced in flashinfer-jit-cache/dist/" >&2
  exit 1
fi

wheel_name=$(basename "${wheel_path}")
if [[ "${wheel_name}" != *"+${WHEEL_MARKER}"* ]]; then
  echo "error: built wheel is missing the expected local-version marker '+${WHEEL_MARKER}'" >&2
  echo "built wheel: ${wheel_name}" >&2
  exit 1
fi

# Sanity check: the SM89 sparse-MLA module must be present in the wheel.
if ! python3 - "${wheel_path}" <<'EOF'
import sys, zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as z:
    hits = [n for n in z.namelist() if "sparse_mla_sm89" in n]
if not hits:
    sys.exit(1)
EOF
then
  echo "error: wheel is missing the sparse_mla_sm89 AOT module" >&2
  echo "check that flashinfer/aot.py registers sparse MLA for SM89" >&2
  exit 1
fi

if [[ "$(realpath "$(dirname "${wheel_path}")")" != "$(realpath "${OUTPUT_DIR}")" ]]; then
  cp -f "${wheel_path}" "${OUTPUT_DIR}/"
fi

echo
echo "Built wheel:"
echo "  ${wheel_name}"
echo
echo "Copied to:"
echo "  ${OUTPUT_DIR}/${wheel_name}"
