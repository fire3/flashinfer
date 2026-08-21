#!/usr/bin/env bash
set -euo pipefail

# Prepare pinned third-party dependencies (git submodules) for the current
# checkout — the FlashInfer equivalent of vLLM's scripts/build/prepare_deps.sh.
#
# FlashInfer pins dependency versions as git submodule commits recorded in the
# index (see `git submodule status`), unlike vLLM which resolves CMake pins.
# Run this after switching branches or fetching upstream so 3rdparty/ matches
# the current checkout.
#
# Usage:
#   bash scripts/build/sm89_prepare_deps.sh          # cutlass, spdlog, cccl
#   bash scripts/build/sm89_prepare_deps.sh --all    # + 3rdparty/nixl (EP)
#   bash scripts/build/sm89_prepare_deps.sh --check  # verify without changing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SUBMODULES=(3rdparty/cutlass 3rdparty/spdlog 3rdparty/cccl)
MODE="update"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) SUBMODULES+=(3rdparty/nixl); shift ;;
    --check) MODE="check"; shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

check_submodules() {
  local bad=0
  for sub in "${SUBMODULES[@]}"; do
    line="$(git submodule status "${sub}")"
    if [[ -z "${line}" || "${line:0:1}" == "-" ]]; then
      echo "missing/not initialized: ${sub}" >&2
      bad=1
    elif [[ "${line:0:1}" == "+" ]]; then
      echo "checkout differs from pinned commit: ${sub}" >&2
      bad=1
    fi
  done
  if [[ "${bad}" -ne 0 ]]; then
    echo "error: run without --check to sync submodules" >&2
    return 1
  fi
  return 0
}

if [[ "${MODE}" == "check" ]]; then
  echo "==> Checking submodule pins"
  check_submodules
  echo "OK: all required submodules match the pinned commits"
  exit 0
fi

echo "==> Initializing submodules at pinned commits: ${SUBMODULES[*]}"
git submodule update --init -- "${SUBMODULES[@]}"

echo "==> Verifying required headers are present"
for f in 3rdparty/cutlass/include 3rdparty/spdlog/include 3rdparty/cccl/cub; do
  if [[ ! -e "${f}" ]]; then
    echo "error: missing ${f}; submodule content did not materialize" >&2
    exit 1
  fi
done

echo "==> JIT include path note"
echo "    With an editable install, flashinfer/data/cccl maps to 3rdparty/cccl"
echo "    (pyproject.toml package-dir), so the JIT include path works once the"
echo "    submodule is initialized. Non-editable wheels carry data/cccl inside."

check_submodules
echo "OK: third-party dependencies are ready (${SUBMODULES[*]})"
