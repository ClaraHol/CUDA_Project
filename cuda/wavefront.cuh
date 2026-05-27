#pragma once

#include "rng.cuh"
#include "types.cuh"
#include <cuda_runtime.h>

// Per-pixel wavefront buffers.
// The host iterates over samples_per_pixel on the outer loop, launching
// init+bounce*max_depth once per sample. All buffers are sized to
// total_pixels = width * height. The accum buffer persists across all sample
// launches and is only zeroed once (via launch_wavefront_accum_clear).
struct WavefrontBuffers {
  Ray *rays;          // current ray for this pixel's in-flight path
  float3 *throughput; // path throughput, reset to (1,1,1) each sample
  bool *active;       // false once the path misses or is absorbed
  float3 *accum;      // per-pixel radiance sum across all samples
  RngState *rng;      // per-pixel RNG state (advanced each sample)

  int total_pixels; // width * height
};

inline cudaError_t allocate_wavefront_buffers(WavefrontBuffers &b, int width,
                                              int height) {
  b.total_pixels = width * height;
  const int n = b.total_pixels;

  auto alloc = [&](auto **ptr, size_t bytes) -> cudaError_t {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess)
      *ptr = nullptr;
    return err;
  };

  cudaError_t err;
  if ((err = alloc(&b.rays, n * sizeof(Ray))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.throughput, n * sizeof(float3))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.active, n * sizeof(bool))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.accum, n * sizeof(float3))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.rng, n * sizeof(RngState))) != cudaSuccess)
    return err;
  return cudaSuccess;
}

inline void free_wavefront_buffers(WavefrontBuffers &b) {
  cudaFree(b.rays);
  b.rays = nullptr;
  cudaFree(b.throughput);
  b.throughput = nullptr;
  cudaFree(b.active);
  b.active = nullptr;
  cudaFree(b.accum);
  b.accum = nullptr;
  cudaFree(b.rng);
  b.rng = nullptr;
}
