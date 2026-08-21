"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, current_compilation_context


def _sparse_mla_supported_cuda_archs() -> list[tuple[int, str]]:
    supported_cuda_archs = [
        (major, minor)
        for major, minor in current_compilation_context.TARGET_CUDA_ARCHS
        if major == 12 or (major == 8 and minor == "9")
    ]
    if not supported_cuda_archs:
        raise RuntimeError(
            "No supported CUDA architectures found for sparse MLA SM89/SM12x."
        )
    return sorted(supported_cuda_archs)


def _format_sparse_mla_arch_tag(major: int, minor: str) -> str:
    return f"sm{major}{minor}"


def _sparse_mla_module_name(
    supported_cuda_archs: list[tuple[int, str]],
) -> str:
    arch_suffix = "_".join(
        _format_sparse_mla_arch_tag(major, minor)
        for major, minor in supported_cuda_archs
    )
    return f"sparse_mla_{arch_suffix}"


def _sparse_mla_nvcc_flags(
    supported_cuda_archs: list[tuple[int, str]],
) -> list[str]:
    return [
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        for major, minor in supported_cuda_archs
    ] + current_compilation_context.COMMON_NVCC_FLAGS


def gen_mla_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11], map_sm107_to_100f=True
    )
    return gen_jit_spec(
        "mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_mla.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mla_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_sparse_mla_sm120_module() -> JitSpec:
    """Sparse-MLA paged attention for SM120.

    Monolithic module: runtime dispatch on model type, head count, top-k,
    page block size, and optional extra page block size happens inside the
    orchestrator.
    """
    supported_cuda_archs = _sparse_mla_supported_cuda_archs()
    nvcc_flags = _sparse_mla_nvcc_flags(supported_cuda_archs)
    return gen_jit_spec(
        _sparse_mla_module_name(supported_cuda_archs),
        [
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_decode_dsv3_2.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_decode_dsv4.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_prefill.cu",
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120_jit_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
