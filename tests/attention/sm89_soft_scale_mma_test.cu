// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone correctness test for the SM89 software-scale FP8 MMA path.
//
// What it validates
// ---------------
// mma_fp8_block_scaled_m16n8k32() in
//   include/flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh
// is compiled for SM89 (__CUDA_ARCH__ < 900 => SPARSE_MLA_USE_SM89_PRIMS=1)
// as *plain* mma.sync.aligned.m16n8k32 (no hardware block scaling) followed by
// a per-accumulator software multiply of the two ue8m0 block scales. This test
// checks, with a torch-free / dependency-free CUDA harness:
//
//   1. multiply_ue8m0() against a double-precision reference over all 256x256
//      scale-byte pairs, covering normal / denormal / overflow / NaN paths.
//   2. The raw m16n8k32 FP8 mma against a double-precision CPU matmul
//      reference (validates the e4m3 A/B fragment packing + accumulator layout).
//   3. The software-scale block mma against
//        D[r][c] = C[r][c] + (sum_k A[r][k]*B[k][c]) * ue8m0(scaleA[r]) * ue8m0(scaleB[c])
//      i.e. the exact semantics the hardware block-scale instruction would provide,
//      reproduced with a plain mma on SM89.
//
// The scale-per-row (A) / scale-per-column (B) association used by the test matches
// the lane distribution the kernel relies on in decode_dsv4_kernel.cuh /
// prefill_kernel.cuh (see the lane-mapping comments in main()).
//
// Build (SM89 AD A-class):
//   nvcc -O2 -arch=sm_89 -std=c++17 \
//        -I flashinfer/include/flashinfer/attention/sparse_mla_sm120 \
//        -o sm89_soft_scale_mma_test sm89_soft_scale_mma_test.cu
// Run:  ./sm89_soft_scale_mma_test
//
// Exit code 0 = all checks pass; any non-zero = failure with a message.

#include "mma_sm120.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// Device-under-test entry points (the code being verified comes from mma_sm120.cuh).
// ---------------------------------------------------------------------------
__device__ float dev_multiply_ue8m0(uint32_t a, uint32_t b) { return multiply_ue8m0(a, b); }

struct MmaArgs {
  uint32_t a0, a1, a2, a3, b0, b1;
};

__global__ void k_multiply_ue8m0(const uint8_t* a, const uint8_t* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dev_multiply_ue8m0(a[i], b[i]);
}

__global__ void k_plain_mma(const MmaArgs* args, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    MmaArgs p = args[i];
    MmaFp8Result r = mma_fp8_m16n8k32(p.a0, p.a1, p.a2, p.a3, p.b0, p.b1, 0.f, 0.f, 0.f, 0.f);
    int* o = reinterpret_cast<int*>(out + (size_t)i * 4);
    o[0] = __float_as_int(r.d0);
    o[1] = __float_as_int(r.d1);
    o[2] = __float_as_int(r.d2);
    o[3] = __float_as_int(r.d3);
  }
}

// Each block = one warp (32 lanes) = one m16n8k32 instance.
__global__ void k_soft_scale_mma(const uint8_t* a_img, const uint8_t* b_img,
                                const uint8_t* sa_img, const uint8_t* sb_img, const float* c_img,
                                float* plain_out, float* soft_out) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;   // groupID 0..7
  const int t = lane & 3;    // threadID_in_group 0..3

  const uint8_t* a = a_img + blockIdx.x * (16 * 32);
  const uint8_t* b = b_img + blockIdx.x * (32 * 8);

  // e4m3 fragment packing, mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3
  // (PTX ISA 9.7.15.5.10):
  //   a0 = A[g][4t..4t+3]        a1 = A[g+8][4t..4t+3]
  //   a2 = A[g][4t+16..4t+19]    a3 = A[g+8][4t+16..4t+19]
  //   b0 = B[4t..4t+3][g]         b1 = B[4t+16..4t+19][g]
  const int kt = 4 * t;
  uint32_t a0 = a[g * 32 + kt] | (a[g * 32 + kt + 1] << 8) | (a[g * 32 + kt + 2] << 16) |
                (a[g * 32 + kt + 3] << 24);
  uint32_t a1 = a[(g + 8) * 32 + kt] | (a[(g + 8) * 32 + kt + 1] << 8) |
                (a[(g + 8) * 32 + kt + 2] << 16) | (a[(g + 8) * 32 + kt + 3] << 24);
  uint32_t a2 = a[g * 32 + kt + 16] | (a[g * 32 + kt + 17] << 8) | (a[g * 32 + kt + 18] << 16) |
                (a[g * 32 + kt + 19] << 24);
  uint32_t a3 = a[(g + 8) * 32 + kt + 16] | (a[(g + 8) * 32 + kt + 17] << 8) |
                (a[(g + 8) * 32 + kt + 18] << 16) | (a[(g + 8) * 32 + kt + 19] << 24);
  uint32_t b0 = b[kt * 8 + g] | (b[(kt + 1) * 8 + g] << 8) | (b[(kt + 2) * 8 + g] << 16) |
                (b[(kt + 3) * 8 + g] << 24);
  uint32_t b1 = b[(kt + 16) * 8 + g] | (b[(kt + 17) * 8 + g] << 8) | (b[(kt + 18) * 8 + g] << 16) |
                (b[(kt + 19) * 8 + g] << 24);

#define WIDX(off) (blockIdx.x * 128 + off)
  MmaFp8Result pr = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, 0.f, 0.f, 0.f, 0.f);
  float* po = plain_out + (size_t)blockIdx.x * 128;
  // Accumulator layout (.f32): d0=(g,2t) d1=(g,2t+1) d2=(g+8,2t) d3=(g+8,2t+1)
  po[g * 8 + 2 * t] = pr.d0;
  po[g * 8 + 2 * t + 1] = pr.d1;
  po[(g + 8) * 8 + 2 * t] = pr.d2;
  po[(g + 8) * 8 + 2 * t + 1] = pr.d3;

  // Per-lane scale regs as the caller kernels lay them out so the shfl inside
  // mma_fp8_block_scaled_m16n8k32 feeds each accumulator its own
  // A-row (sa0/sa1) x B-column (sb0/sb1) scale:
  //   A row r   -> lane 4r  (r<8) or lane 4r+1 (r>=8)
  //   B column c -> lane 4c
  const uint8_t* sa = sa_img + blockIdx.x * 16;
  const uint8_t* sc = sb_img + blockIdx.x * 8;
  int arow = g + ((lane & 1) * 8);
  uint8_t sfa = sa[arow];
  uint8_t sfb = ((lane & 3) == 0) ? sc[lane >> 2] : (uint8_t)0;

  float c0 = c_img[WIDX(g * 8 + 2 * t)];
  float c1 = c_img[WIDX(g * 8 + 2 * t + 1)];
  float c2 = c_img[WIDX((g + 8) * 8 + 2 * t)];
  float c3 = c_img[WIDX((g + 8) * 8 + 2 * t + 1)];

  MmaFp8Result sr = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);
  float* so = soft_out + (size_t)blockIdx.x * 128;
  so[g * 8 + 2 * t] = sr.d0;
  so[g * 8 + 2 * t + 1] = sr.d1;
  so[(g + 8) * 8 + 2 * t] = sr.d2;
  so[(g + 8) * 8 + 2 * t + 1] = sr.d3;
#undef WIDX
}

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------
static double ue8m0_ref(int x) { return ldexp(1.0, x - 127); }

static float deq_e4m3(uint8_t x) {
  __nv_fp8_e4m3 v;
  v.__x = x;
  return (float)v;  // e4m3 bytes (0..15 here) are exact small floats
}

static float multiply_ue8m0_ref(uint8_t a, uint8_t b) {
  if (a == 0xff || b == 0xff) {
    uint32_t nan = 0x7fc00000u;
    float f;
    memcpy(&f, &nan, 4);
    return f;  // quiet NaN
  }
  double v = ue8m0_ref(a) * ue8m0_ref(b);  // 2^(a+b-254)
  // (a+b-254) is the true binary exponent of the product
  if (v >= 0x1p128 || std::isinf(v)) {
    uint32_t inf = 0x7f800000u;
    float f;
    memcpy(&f, &inf, 4);
    return f;
  }
  if (v < 0x1p-149) return 0.0f;  // below smallest subnormal
  // round-to-nearest-even of the (possibly subnormal) power of two to fp32
  return (float)v;
}

static bool close_float(float got, float exp, float tol, bool* exact_out) {
  uint32_t gb, eb;
  memcpy(&gb, &got, 4);
  memcpy(&eb, &exp, 4);
  if (gb == eb) return *exact_out = true, true;
  *exact_out = false;
  if (std::isnan(got) && std::isnan(exp)) return true;  // NaN == NaN (bit differs ok)
  return std::fabs((double)got - (double)exp) <= std::fabs((double)exp) * tol + 1e-5;
}

int main() {
  int dev = 0;
  cudaError_t ec = cudaSetDevice(dev);
  if (ec != cudaSuccess) { printf("no CUDA device: %s\n", cudaGetErrorString(ec)); return 2; }
  int cc_major = 0, cc_minor = 0;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (cc_major < 8) { printf("test requires sm_80+ (got %d.%d)\n", cc_major, cc_minor); return 2; }
#if defined(__CUDA_ARCH__)
  printf("compiled arch=%d\n", __CUDA_ARCH__);
#endif

  int fails = 0;

  // ---- 1. multiply_ue8m0 exhaustive scale-product check ----
  {
    const int n = 256 * 256;
    std::vector<uint8_t> a(n), b(n);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) { a[i] = (uint8_t)(i >> 8); b[i] = (uint8_t)(i & 0xff); }
    uint8_t *da, *db;
    float* dout;
    cudaMalloc(&da, n); cudaMalloc(&db, n); cudaMalloc(&dout, (size_t)n * 4);
    cudaMemcpy(da, a.data(), n, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n, cudaMemcpyHostToDevice);
    k_multiply_ue8m0<<<(n + 255) / 256, 256>>>(da, db, dout, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("multiply kernel launch failed: %s\n", cudaGetErrorString(err)); return 2; }
    cudaMemcpy(out.data(), dout, (size_t)n * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; i++) {
      float exp = multiply_ue8m0_ref(a[i], b[i]);
      bool exact = false;
      if (!close_float(out[i], exp, 0.0f, &exact)) {
        if (bad < 10) printf("  multiply_ue8m0 MISM at a=%d b=%d got=%g exp=%g\n", a[i], b[i], out[i], exp);
        bad++;
      }
    }
    printf("multiply_ue8m0: %d/%d pairs match (%s)\n", n - bad, n, bad ? "FAIL" : "PASS");
    fails += bad ? 1 : 0;
    cudaFree(da); cudaFree(db); cudaFree(dout);
  }

  // ---- 2+3. mma fragment + software-scale block mma ----
  {
    const int N = 256;
    std::vector<uint8_t> a_img((size_t)N * 16 * 32), b_img((size_t)N * 32 * 8);
    std::vector<uint8_t> sa_img((size_t)N * 16), sb_img((size_t)N * 8);
    std::vector<float> c_img((size_t)N * 128);
    std::vector<float> plain_h((size_t)N * 128), soft_h((size_t)N * 128);

    unsigned seed = 12345;
    for (int i = 0; i < N; i++) {
      for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++) {
          seed = seed * 1103515245u + 12345u;
          int v = (int)((seed >> 16) % 16);  // 0..15, exact in e4m3
          a_img[i * 512 + r * 32 + k] = (uint8_t)v;
        }
      for (int k = 0; k < 32; k++)
        for (int c = 0; c < 8; c++) {
          seed = seed * 1103515245u + 12345u;
          int v = (int)((seed >> 16) % 16);
          b_img[i * 256 + k * 8 + c] = (uint8_t)v;
        }
      // scale bytes: mix of 1.0 (0x7f), small/large powers, and edge cases
      // mix of 1.0 (0x7f), small/large powers, underflow (0x00) and NaN (0xff) scales
      const int sa_pool[8] = {0x00, 0x66, 0x7f, 0x80, 0x90, 0x70, 0xfe, 0xff};
      const int sa_pool_n = 8;
      for (int r = 0; r < 16; r++) {
        seed = seed * 1103515245u + 12345u;
        sa_img[i * 16 + r] = (uint8_t)sa_pool[(seed >> 16) % sa_pool_n];
      }
      for (int c = 0; c < 8; c++) {
        seed = seed * 1103515245u + 12345u;
        sb_img[i * 8 + c] = (uint8_t)sa_pool[(seed >> 16) % sa_pool_n];
      }
      for (int idx = 0; idx < 128; idx++) {
        seed = seed * 1103515245u + 12345u;
        c_img[i * 128 + idx] = (float)(((int)((seed >> 16) % 2001) - 1000) / 64.0);
      }
    }

    uint8_t *da, *db, *dsa, *dsb;
    float *dc, *dplain, *dsoft;
    cudaMalloc(&da, a_img.size()); cudaMemcpy(da, a_img.data(), a_img.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&db, b_img.size()); cudaMemcpy(db, b_img.data(), b_img.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dsa, sa_img.size()); cudaMemcpy(dsa, sa_img.data(), sa_img.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dsb, sb_img.size()); cudaMemcpy(dsb, sb_img.data(), sb_img.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dc, c_img.size() * 4); cudaMemcpy(dc, c_img.data(), c_img.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dplain, (size_t)N * 128 * 4);
    cudaMalloc(&dsoft, (size_t)N * 128 * 4);
    k_soft_scale_mma<<<N, 32>>>(da, db, dsa, dsb, dc, dplain, dsoft);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("soft-scale kernel launch failed: %s\n", cudaGetErrorString(err)); return 2; }
    cudaMemcpy(plain_h.data(), dplain, (size_t)N * 128 * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(soft_h.data(), dsoft, (size_t)N * 128 * 4, cudaMemcpyDeviceToHost);

    double max_raw_err = 0.0, max_soft_err = 0.0;
    int raw_bad = 0, soft_bad = 0;
    for (int i = 0; i < N; i++) {
      for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 8; c++) {
          // independent fp64 matmul reference
          double sum = 0.0;
          for (int k = 0; k < 32; k++)
            sum += (double)deq_e4m3(a_img[i * 512 + r * 32 + k]) * deq_e4m3(b_img[i * 256 + k * 8 + c]);
          float raw = plain_h[i * 128 + r * 8 + c];
          double raw_err = std::fabs((double)raw - sum);
          if (raw_err > max_raw_err) max_raw_err = raw_err;
          if (raw_err > 0.01) raw_bad++;

          float mul = (float)(ue8m0_ref(sa_img[i * 16 + r]) * ue8m0_ref(sb_img[i * 8 + c]));
          double expect = (double)c_img[i * 128 + r * 8 + c] + sum * (double)mul;
          float got = soft_h[i * 128 + r * 8 + c];
          double soft_err = std::fabs((double)got - expect);
          // tolerance: |c| + |sum*mul| can be ~2000*4096 ~ 8e6; allow 1e-3 relative
          double tol = 1e-3 * (std::fabs((double)c_img[i * 128 + r * 8 + c]) + std::fabs(sum * mul) + 1.0);
          if (soft_err > tol) {
            if (soft_bad < 10)
              printf("  soft mma MISM i=%d r=%d c=%d got=%g expect=%g err=%g tol=%g raw=%g sa=%d sb=%d\n", i, r, c, got, expect, soft_err, tol, raw, sa_img[i*16+r], sb_img[i*8+c]);
            soft_bad++;
          } else if (soft_err > max_soft_err)
            max_soft_err = soft_err / tol;
        }
      }
    }
    printf("plain mma vs fp64 matmul: %d/%d bad, max_abs_err=%g (%s)\n", raw_bad, N * 128, max_raw_err, raw_bad ? "FAIL" : "PASS");
    printf("soft-scale mma vs ref:     %d/%d bad, max_rel_to_tol=%g (%s)\n", soft_bad, N * 128, max_soft_err, soft_bad ? "FAIL" : "PASS");
    fails += raw_bad ? 1 : 0;
    fails += soft_bad ? 1 : 0;
  }

  if (fails == 0) printf("ALL TESTS PASSED\n");
  else printf("%d TEST GROUP(S) FAILED\n", fails);
  return fails ? 1 : 0;
}
