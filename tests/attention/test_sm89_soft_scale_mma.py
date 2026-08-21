# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifies the SM89 software-scale FP8 MMA emulation used by sparse MLA on Ada.

On SM89 (and any < sm_90) ``SPARSE_MLA_USE_SM89_PRIMS`` makes
``mma_fp8_block_scaled_m16n8k32`` fall back to a plain
``mma.m16n8k32`` plus a software multiply of the two ue8m0 block scales
(include/_arch_/mma_sm120.cuh). This test compiles the self-contained
harness ``sm89_soft_scale_mma_test.cu`` (which #includes the real header)
and asserts:

  1. ``multiply_ue8m0`` matches an exhaustive double-precision reference over all
     256x256 scale-byte pairs (normal / denormal / overflow / NaN).
  2. the raw m16n8k32 e4m3 mma matches a double-precision matmul (this
     also pins the e4m3 fragment and accumulator layouts).
  3. the software-scale path yields
       D[r][c] = C[r][c] + sum_k(A[r][k]B[k][c])*ue8m0(scaleA[r])*ue8m0(scaleB[c])
     i.e. the exact semantics the hardware block-scale mma would provide.

Skipped unless an SM89 (8.x) GPU and nvcc are available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from flashinfer.utils import get_compute_capability

_CU = Path(__file__).parent / "sm89_soft_scale_mma_test.cu"
_ARCH_INC = (
    Path(__file__).resolve().parents[2] / "include" / "flashinfer" / "attention" / "sparse_mla_sm120" / "arch"
)


def _find_nvcc():
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda:
        cand = Path(cuda) / "bin" / "nvcc"
        if cand.exists():
            return str(cand)
    for root in ("/usr/local",):
        for d in sorted(Path(root).glob("cuda-*"), reverse=True):
            cand = d / "bin" / "nvcc"
            if cand.exists():
                return str(cand)
    return None


def _is_sm8x() -> bool:
    if not _CU.exists():
        return False
    if not torch_available():
        return False
    import torch

    return torch.cuda.is_available() and get_compute_capability(torch.device("cuda"))[0] == 8


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not (_is_sm8x() and _find_nvcc()),
    reason="Requires an SM8x GPU and nvcc to compile/run the CUDA harness.",
)
def test_sm89_soft_scale_mma(tmp_path):
    nvcc = _find_nvcc()
    assert nvcc
    exe = tmp_path / "sm89_soft_scale_mma_test"
    proc = subprocess.run(
        [
            nvcc,
            "-O2",
            "-arch=sm_89",
            "-std=c++17",
            f"-I{_ARCH_INC}",
            "-o",
            str(exe),
            str(_CU),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"nvcc failed:\n{proc.stderr}"
    run = subprocess.run([str(exe)], cwd=str(tmp_path), capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"harness failed rc={run.returncode}\n{run.stdout}\n{run.stderr}"
    assert "ALL TESTS PASSED" in run.stdout
