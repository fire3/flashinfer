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

// cp.async: global → shared async copy

__device__ __forceinline__ void cp_async_4B(void* smem_ptr, const void* gmem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_8B(void* smem_ptr, const void* gmem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_16B_l2(void* smem_ptr, const void* gmem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
#if SPARSE_MLA_USE_SM89_PRIMS
  // L2::128B hint is illegal on SM89 (illegal memory access); plain
  // cp.async is functionally equivalent (just loses the cache hint).
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem_ptr));
#else
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem_ptr));
#endif
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void cp_async_mbarrier_arrive_noinc(uint64_t* mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(addr) : "memory");
}

// cp.async.mbarrier.arrive: binds every prior cp.async copy issued by THIS
// thread to the mbarrier — upon completion of those copies the barrier's
// pending count is decremented (an arrive-on). Without .noinc the pending
// count is first incremented by 1, so the operation is zero-net on the
// count: it is a *visibility carrier* for the copies, not the completing
// arrival. The phase still completes via one thread's plain mbarrier_arrive
// (release), matching PTX ISA Example 1 — the mbarrier init count must only
// account for that completing arrival.
//
// This is the SM80/89 substitute for SM90+ TMA tx-count tracking: PTX ISA
// §mbarrier.test_wait/try_wait acquire-ordering item 2 guarantees that all
// cp.async operations requested prior to cp.async.mbarrier.arrive are
// performed and made visible to the acquiring waiter. A plain release
// mbarrier.arrive alone does NOT cover cp.async writes (they are async-proxy
// operations, excluded from item 1), and fence.proxy.async requires sm_90+.
__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::"r"(addr) : "memory");
}

// cp.async.bulk: SM90+ bulk global → shared (mbarrier-based completion)
__device__ __forceinline__ void cp_async_bulk_g2s(void* smem_dst, const void* gmem_src,
                                                  uint32_t bytes, uint64_t* mbar) {
#if SPARSE_MLA_USE_SM89_PRIMS
  // SM89: no cp.async.bulk / expect_tx. Emulate with 16B cp.async chunks.
  // Callers must follow the gather with cp_async_mbarrier_arrive() (per
  // issuing thread) + one mbarrier_arrive() so the math side's acquire sees
  // the copies (see kv_cache_io.cuh / decode_dsv4 issue_gather).
  (void)mbar;
#pragma unroll
  for (uint32_t off = 0; off < bytes; off += 16) {
    cp_async_16B(static_cast<uint8_t*>(smem_dst) + off,
                 static_cast<const uint8_t*>(gmem_src) + off);
  }
#else
  uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];\n" ::"r"(dst_addr),
      "l"(gmem_src), "r"(bytes), "r"(mbar_addr));
#endif
}

// cp.async.bulk with L2 cache hint (evict_first for streaming KV data)
__device__ __forceinline__ void cp_async_bulk_g2s_l2hint(void* smem_dst, const void* gmem_src,
                                                         uint32_t bytes, uint64_t* mbar,
                                                         uint64_t cache_policy) {
#if SPARSE_MLA_USE_SM89_PRIMS
  // SM89: same 16B cp.async emulation as cp_async_bulk_g2s; the L2 hint has
  // no SM89 form (see cp_async_16B_l2), so it is dropped.
  (void)mbar;
  (void)cache_policy;
#pragma unroll
  for (uint32_t off = 0; off < bytes; off += 16) {
    cp_async_16B_l2(static_cast<uint8_t*>(smem_dst) + off,
                    static_cast<const uint8_t*>(gmem_src) + off);
  }
#else
  uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1], %2, [%3], %4;\n" ::"r"(dst_addr),
      "l"(gmem_src), "r"(bytes), "r"(mbar_addr), "l"(cache_policy));
#endif
}

// Create L2 evict-first cache policy (streaming data consumed once)
__device__ __forceinline__ uint64_t create_l2_evict_first_policy() {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

// Store 8 floats (256-bit) with L2::evict_last hint.
// L2::evict_last requires .v8.b32 or .v4.b64 — 128-bit stores are not supported.
__device__ __forceinline__ void store_8f_evict_last(float* addr, float4 v0, float4 v1) {
  asm volatile(
      "st.global.L2::evict_last.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"l"(addr),
      "r"(__float_as_int(v0.x)), "r"(__float_as_int(v0.y)), "r"(__float_as_int(v0.z)),
      "r"(__float_as_int(v0.w)), "r"(__float_as_int(v1.x)), "r"(__float_as_int(v1.y)),
      "r"(__float_as_int(v1.z)), "r"(__float_as_int(v1.w))
      : "memory");
}
