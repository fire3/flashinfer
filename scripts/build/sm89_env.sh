#!/usr/bin/env bash
# Source this file to configure your shell for building/developing FlashInfer
# on SM89 (NVIDIA Ada, e.g. L40S).
#
#   source scripts/build/sm89_env.sh
#
# Every exported variable below can be overridden before sourcing.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# SM89-only build: nvcc emits sm_89 SASS.
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.9}"

# Local-version marker for wheels, e.g. flashinfer_python-0.6.16.post3+sm89.
export FLASHINFER_LOCAL_VERSION="${FLASHINFER_LOCAL_VERSION:-${SM89_WHEEL_MARKER:-sm89}}"

# CCCL opt-out: when the CUDA runtime headers and nvcc disagree on the CUDA
# version (e.g. pip nvidia-cuda-nvcc wheels), CCCL's CTK compatibility check
# can fail the build spuriously. This is the upstream-sanctioned switch and is
# harmless when the versions genuinely match.
export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1}"

# The moe_ep (NIXL/NCCL-EP) backends are not needed for SM89 sparse MLA.
# Disable them by default so `pip install -e .` does not try to build
# 3rdparty/nixl; override with BUILD_NIXL_EP=1 if EP is required.
export BUILD_NIXL_EP="${BUILD_NIXL_EP:-0}"
export BUILD_NCCL_EP="${BUILD_NCCL_EP:-0}"

# nvcc discovery: prefer PATH, then the pip nvidia-cuda-nvcc wheel layout.
if ! command -v nvcc >/dev/null 2>&1; then
  site_pkgs="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
  for cand in "${site_pkgs}/nvidia/cu13/bin" "${site_pkgs}/nvidia/cuda_nvcc/bin"; do
    if [[ -n "${cand}" && -x "${cand}/nvcc" ]]; then
      export PATH="${cand}:${PATH}"
      break
    fi
  done
fi
