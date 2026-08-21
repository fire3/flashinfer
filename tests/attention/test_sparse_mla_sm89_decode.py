# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Ada-specific correctness tests for sparse-MLA DSV4 decode."""

from __future__ import annotations

import pytest
import torch

from flashinfer.mla._sparse_mla_sm120 import (
    _decode_dsv4_dispatchable,
    _sparse_mla_sm120_paged_attention,
)
from flashinfer.utils import get_compute_capability
from tests.attention.test_sparse_mla_sm120 import (
    _make_decode_scratch,
    _ref_sparse_attn,
    dequantize_kv_dsv4,
    quantize_kv_dsv4,
)


def _is_sm89() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) == (8, 9)


pytestmark = pytest.mark.skipif(not _is_sm89(), reason="Test requires SM89.")


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_sparse_mla_sm89_decode_dsv4_distributed_scales(num_tokens: int) -> None:
    """Each MMA accumulator uses its own query-row and KV-column scales."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 16, 128
    d_qk = d_v = 512
    page_block_size = 64
    num_blocks = 8
    num_slots = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks,
            page_block_size,
            1,
            d_qk,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    kv_tile_scales = torch.logspace(
        -1, 1, 7, base=2, device=device, dtype=torch.float32
    )
    for tile, scale in enumerate(kv_tile_scales):
        lo = tile * 64
        kv_bf16[..., lo : lo + 64].mul_(scale)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10
    )
    q_head_scales = torch.logspace(
        -3, 3, num_heads, base=2, device=device, dtype=torch.float32
    )
    q.mul_(q_head_scales.view(1, num_heads, 1))
    indices = torch.randint(
        0, num_slots, (num_tokens, topk), device=device, dtype=torch.int32
    )
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.empty(
        num_tokens, num_heads, d_v, device=device, dtype=torch.bfloat16
    )
    out_lse = torch.empty(num_tokens, num_heads, device=device, dtype=torch.float32)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    assert _decode_dsv4_dispatchable(num_tokens, num_heads, topk, d_qk, page_block_size)
    _sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=3e-3, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=2e-3, rtol=0)


def test_sparse_mla_sm89_decode_dsv4_min_ue8m0_scale() -> None:
    """UE8M0 byte zero is 2^-127 rather than a zero sentinel."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads, topk = 1, 16, 128
    d_qk = d_v = 512
    page_block_size = 64
    num_blocks = 2
    block_bytes = page_block_size * 584

    storage = torch.zeros(num_blocks, block_bytes, device=device, dtype=torch.uint8)
    token_data = storage[:, : page_block_size * 576].view(
        num_blocks, page_block_size, 576
    )
    fp8_max = (
        torch.full((), 448.0, device=device, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .item()
    )
    token_data[..., :448].fill_(fp8_max)
    kv_packed = storage.view(num_blocks, page_block_size, 1, 584)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = torch.zeros(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
    q[..., :448].fill_(2.0**112)
    indices = torch.randint(
        0,
        num_blocks * page_block_size,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )
    sm_scale = d_qk**-0.5
    _, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.empty(
        num_tokens, num_heads, d_v, device=device, dtype=torch.bfloat16
    )
    out_lse = torch.empty_like(ref_lse)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)
    _sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(out_lse, ref_lse, atol=2e-3, rtol=0)
