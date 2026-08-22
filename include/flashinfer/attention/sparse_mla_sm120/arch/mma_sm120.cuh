// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "common.cuh"

// SM120 MMA instruction wrappers.
//
// Standard (no scale):
//   FP8:  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
//   BF16: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Block-scaled (UE8M0 scale applied in hardware, zero overhead):
//   FP8:  mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
//         .row.col.f32.e4m3.e4m3.f32.ue8m0

struct MmaFp8Result {
  float d0, d1, d2, d3;
};
struct MmaBf16Result {
  float d0, d1, d2, d3;
};

__device__ __forceinline__ float multiply_ue8m0(uint32_t a, uint32_t b) {
  const int biased_exp = static_cast<int>(a) + static_cast<int>(b) - 127;
  if (__builtin_expect(a != 0xff && b != 0xff && biased_exp > 0 && biased_exp < 255, 1)) {
    return __uint_as_float(static_cast<uint32_t>(biased_exp) << 23);
  }
  if (a == 0xff || b == 0xff) return __uint_as_float(0x7fc00000);
  if (biased_exp >= 255) return __uint_as_float(0x7f800000);
  if (biased_exp < -22) return 0.f;
  return __uint_as_float(1u << (biased_exp + 22));
}

__device__ __forceinline__ MmaFp8Result mma_fp8_m16n8k32(uint32_t a0, uint32_t a1, uint32_t a2,
                                                         uint32_t a3, uint32_t b0, uint32_t b1,
                                                         float c0, float c1, float c2, float c3) {
  MmaFp8Result r;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}

#if SPARSE_MLA_USE_SM89_PRIMS
// ── SM89-only software block-scale path ─────────────────────────────────────
// Combines the per-lane ue8m0 row/column scales into the four per-accumulator
// factors. The shuffles below are the only warp-wide traffic of the
// software-scale path, so prepare once per (scale_a, scale_b) and reuse across
// K/N tiles.
//
// Lane contract (see tests/attention/sm89_soft_scale_mma_test.cu):
//   scale_a(lane) = ue8m0(A row (lane>>2) + ((lane & 1) << 3))   -- valid on all lanes
//   scale_b(lane) = ue8m0(B column (lane>>2))                    -- sb0/sb1 read tid==0 lanes
struct MmaFp8Scale {
  // s00=(row g, col 2t), s01=(row g, col 2t+1), s10=(row g+8, col 2t),
  // s11=(row g+8, col 2t+1), matching the m16n8k32 accumulator layout.
  float s00, s01, s10, s11;
};

__device__ __forceinline__ MmaFp8Scale prepare_block_scale(uint8_t scale_a, uint8_t scale_b) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t quad_base = lane & ~3u;
  const uint32_t col_pair = lane & 3u;

  // Every lane already holds the scale of one of the two rows its accumulator
  // touches, so one shuffle (+ select) replaces the two A shuffles.
  const uint32_t sa_own = static_cast<uint32_t>(scale_a);
  const uint32_t sa_peer = __shfl_sync(0xffffffffu, sa_own, quad_base + ((lane & 1u) ^ 1u));
  const uint32_t sa0 = (lane & 1u) ? sa_peer : sa_own;
  const uint32_t sa1 = (lane & 1u) ? sa_own : sa_peer;

  const uint32_t sb0 = __shfl_sync(0xffffffffu, static_cast<uint32_t>(scale_b), col_pair * 8u);
  const uint32_t sb1 =
      __shfl_sync(0xffffffffu, static_cast<uint32_t>(scale_b), col_pair * 8u + 4u);

  return {multiply_ue8m0(sa0, sb0), multiply_ue8m0(sa0, sb1), multiply_ue8m0(sa1, sb0),
          multiply_ue8m0(sa1, sb1)};
}

// Block-scaled MMA with a precomputed scale pair (SM89 only). Accumulates the
// plain m16n8k32 product scaled by the four factors into C.
__device__ __forceinline__ MmaFp8Result mma_fp8_block_scaled_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, float c0,
    float c1, float c2, float c3, const MmaFp8Scale& scale) {
  MmaFp8Result mma = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);
  return {fmaf(mma.d0, scale.s00, c0), fmaf(mma.d1, scale.s01, c1), fmaf(mma.d2, scale.s10, c2),
          fmaf(mma.d3, scale.s11, c3)};
}
#endif

__device__ __forceinline__ MmaFp8Result mma_fp8_block_scaled_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, float c0,
    float c1, float c2, float c3, uint8_t scale_a, uint8_t scale_b) {
#if SPARSE_MLA_USE_SM89_PRIMS
  return mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3,
                                       prepare_block_scale(scale_a, scale_b));
#else
  MmaFp8Result r;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32"
      ".row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(static_cast<uint32_t>(scale_a)), "n"(static_cast<uint16_t>(0)),
        "n"(static_cast<uint16_t>(0)), "r"(static_cast<uint32_t>(scale_b)),
        "n"(static_cast<uint16_t>(0)), "n"(static_cast<uint16_t>(0)));
  return r;
#endif
}

__device__ __forceinline__ MmaBf16Result mma_bf16_m16n8k16(uint32_t a0, uint32_t a1, uint32_t a2,
                                                           uint32_t a3, uint32_t b0, uint32_t b1,
                                                           float c0, float c1, float c2, float c3) {
  MmaBf16Result r;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "=f"(r.d0), "=f"(r.d1), "=f"(r.d2), "=f"(r.d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  return r;
}
