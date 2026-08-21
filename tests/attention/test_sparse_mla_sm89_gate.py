"""Tests for sparse MLA SM89 JIT and runtime wiring."""

from __future__ import annotations

import torch


def test_sparse_mla_jit_spec_targets_sm89(monkeypatch) -> None:
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "8.9 12.0f")

    from flashinfer.compilation_context import CompilationContext
    import flashinfer.jit.mla as jit_mla

    monkeypatch.setattr(
        jit_mla,
        "current_compilation_context",
        CompilationContext(),
    )

    spec = jit_mla.gen_sparse_mla_sm120_module()
    flags = " ".join(spec.extra_cuda_cflags or [])

    assert spec.name == "sparse_mla_sm89_sm120f"
    assert "-gencode=arch=compute_89,code=sm_89" in flags
    assert "-gencode=arch=compute_120f,code=sm_120f" in flags


def test_sparse_mla_jit_spec_sm89_only_name(monkeypatch) -> None:
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "8.9")

    from flashinfer.compilation_context import CompilationContext
    import flashinfer.jit.mla as jit_mla

    monkeypatch.setattr(
        jit_mla,
        "current_compilation_context",
        CompilationContext(),
    )

    spec = jit_mla.gen_sparse_mla_sm120_module()
    flags = " ".join(spec.extra_cuda_cflags or [])

    assert spec.name == "sparse_mla_sm89"
    assert "-gencode=arch=compute_89,code=sm_89" in flags


def test_sparse_mla_runtime_gate_admits_sm89() -> None:
    from flashinfer.mla._sparse_mla_sm120 import (
        _SparseMLAPagedAttentionRunner,
        _sparse_mla_sm120_paged_attention,
        sparse_mla_sm120_decode_dsv3_2,
        sparse_mla_sm120_decode_dsv4,
    )

    gated_callables = (
        _sparse_mla_sm120_paged_attention,
        _SparseMLAPagedAttentionRunner.__init__,
        sparse_mla_sm120_decode_dsv3_2,
        sparse_mla_sm120_decode_dsv4,
    )

    for fn in gated_callables:
        assert fn.is_compute_capability_supported(89)
        assert not fn.is_compute_capability_supported(86)


def test_batch_decode_mla_backend_resolver_prefers_sparse_on_sm89(monkeypatch) -> None:
    from flashinfer.mla._core import _resolve_batch_decode_mla_backend
    from flashinfer.mla import _core

    monkeypatch.setattr(_core, "get_compute_capability", lambda _device: (12, 0))
    backend = _resolve_batch_decode_mla_backend(
        torch.device("cpu"),
        requested_backend="auto",
        sparse_mla_top_k=128,
    )
    assert backend == "sparse"

    monkeypatch.setattr(_core, "get_compute_capability", lambda _device: (8, 9))
    backend = _resolve_batch_decode_mla_backend(
        torch.device("cpu"),
        requested_backend="auto",
        sparse_mla_top_k=128,
    )
    assert backend == "sparse"

    monkeypatch.setattr(_core, "get_compute_capability", lambda _device: (12, 0))
    backend = _resolve_batch_decode_mla_backend(
        torch.device("cpu"),
        requested_backend="auto",
        sparse_mla_top_k=0,
    )
    assert backend == "xqa"


def test_batch_decode_with_kv_cache_mla_auto_routes_sm89_sparse(monkeypatch) -> None:
    from flashinfer.mla import _core

    sentinel = object()

    def _fake_sparse(**kwargs):
        assert kwargs["sparse_mla_top_k"] == 128
        return sentinel

    monkeypatch.setattr(_core, "get_compute_capability", lambda _device: (8, 9))
    monkeypatch.setattr(_core, "_trtllm_batch_decode_sparse_mla_v32_sm120", _fake_sparse)

    result = _core.trtllm_batch_decode_with_kv_cache_mla(
        query=torch.empty(1, 1, 1, 576, dtype=torch.bfloat16),
        kv_cache=torch.empty(1, 1, 656, dtype=torch.uint8),
        workspace_buffer=torch.empty(1, dtype=torch.uint8),
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=torch.empty(1, 1, 128, dtype=torch.int32),
        seq_lens=None,
        max_seq_len=1,
        sparse_mla_top_k=128,
        backend="auto",
    )

    assert result is sentinel
