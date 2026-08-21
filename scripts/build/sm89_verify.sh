#!/usr/bin/env bash
set -euo pipefail

# Verify the SM89 sparse-MLA build: run the gate/decode tests and smoke-check
# the installed package / built wheel.
#
# Usage:
#   bash scripts/build/sm89_verify.sh                # pytest + import smoke
#   bash scripts/build/sm89_verify.sh --wheel <path> # also check a built wheel
#
# Optional env vars:
#   PYTEST_ARGS   extra args for pytest (e.g. "-k decode -x")
#
# Note: the gate tests run without a GPU, but test_sparse_mla_sm89_decode.py
# requires an SM89 GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WHEEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel) WHEEL="${2:-}"; shift 2 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "==> Import smoke test"
python3 - <<'EOF'
import flashinfer
print("flashinfer:", flashinfer.__file__)
print("version:", getattr(flashinfer, "__version__", "unknown"))
EOF

echo "==> Running SM89 sparse-MLA tests"
python3 -m pytest \
  tests/attention/test_sparse_mla_sm89_gate.py \
  tests/attention/test_sparse_mla_sm89_decode.py \
  ${PYTEST_ARGS:-}

if [[ -n "${WHEEL}" ]]; then
  echo "==> Checking wheel ${WHEEL} for the sparse_mla_sm89 AOT module"
  python3 - "${WHEEL}" <<'EOF'
import sys, zipfile
wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as z:
    hits = [n for n in z.namelist() if "sparse_mla_sm89" in n]
if not hits:
    print("error: wheel is missing the sparse_mla_sm89 AOT module")
    sys.exit(1)
print("found:", ", ".join(hits[:3]))
EOF
fi

echo "OK: SM89 sparse-MLA verification passed"
