#!/usr/bin/env bash
set -euo pipefail

# Build a FlashInfer wheel explicitly marked for SM89-only use (local version
# marker, e.g. flashinfer_python-0.6.16.post3+sm89-...).
#
# Usage:
#   bash scripts/build/sm89_build_wheel.sh            # AOT wheel into dist/
#   bash scripts/build/sm89_build_wheel.sh --editable # dev: pip install -e .
#
# Optional env vars: OUTPUT_DIR, SM89_WHEEL_MARKER, MAX_JOBS,
#                    FLASHINFER_NVCC_THREADS (see sm89_env.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source scripts/build/sm89_env.sh

if [[ "${FLASHINFER_CUDA_ARCH_LIST}" != "8.9" ]]; then
  echo "error: SM89 wheel requires FLASHINFER_CUDA_ARCH_LIST=8.9 (got ${FLASHINFER_CUDA_ARCH_LIST})" >&2
  exit 1
fi

if ! bash scripts/build/sm89_prepare_deps.sh --check >/dev/null 2>&1; then
  echo "error: third-party deps not ready; run: bash scripts/build/sm89_prepare_deps.sh" >&2
  exit 1
fi

if [[ "${1:-}" == "--editable" ]]; then
  echo "==> Installing flashinfer-python in editable mode (--no-build-isolation)"
  python3 -m pip install -e . --no-build-isolation
  echo "==> Done. Editable install active."
  exit 0
fi

echo "=========================================="
echo "Building FlashInfer SM89 wheel"
echo "=========================================="
echo "Python: $(python3 --version)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "CUDA arch list: ${FLASHINFER_CUDA_ARCH_LIST}"
echo "Wheel marker: ${FLASHINFER_LOCAL_VERSION}"
echo "MAX_JOBS: ${MAX_JOBS:-unset}"
echo "FLASHINFER_NVCC_THREADS: ${FLASHINFER_NVCC_THREADS:-unset}"
echo

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
mkdir -p "${OUTPUT_DIR}"

python3 -m pip install --upgrade build
rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}/dist" "${REPO_ROOT}"/*.egg-info
python3 -m build --wheel --no-isolation

wheel_path=$(find "${REPO_ROOT}/dist" -maxdepth 1 -name "*.whl" | head -n 1)
if [[ -z "${wheel_path}" ]]; then
  echo "error: wheel build finished but no wheel was produced in dist/" >&2
  exit 1
fi

wheel_name=$(basename "${wheel_path}")
if [[ "${wheel_name}" != *"+${FLASHINFER_LOCAL_VERSION}"* ]]; then
  echo "error: built wheel is missing the expected local-version marker '+${FLASHINFER_LOCAL_VERSION}'" >&2
  echo "built wheel: ${wheel_name}" >&2
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
