#pragma once

#include "rng.cuh"
#include "types.cuh"
#include <cuda_runtime.h>

struct WavefrontBuffers {
  Ray *rays;          // current ray for this pixel's in-flight path
  float3 *throughput; // path throughput, reset to (1,1,1) each sample
  bool *active;       // false once the path misses or is absorbed
  float3 *accum;      // per-pixel radiance sum across all samples
  RngState *rng;      // per-pixel RNG state (advanced each sample)
  int *pixel_index;   // pointer to gpu int array

  int total_paths;  // width * height * samples_per_pixel
  int total_pixels; // width * height
};

inline cudaError_t allocate_wavefront_buffers(WavefrontBuffers &b, int width,
                                              int height,
                                              int samples_per_pixel) {
  b.total_paths = width * height * samples_per_pixel;
  b.total_pixels = width * height;

  // rays, throughput, active, rng all sized to total_paths
  // accum sized to total_pixels

  const int np = b.total_paths;   // per-path buffers
  const int npx = b.total_pixels; // per-pixel buffers

  auto alloc = [&](auto **ptr, size_t bytes) -> cudaError_t {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess)
      *ptr = nullptr;
    return err;
  };

  cudaError_t err;
  if ((err = alloc(&b.rays, np * sizeof(Ray))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.throughput, np * sizeof(float3))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.active, np * sizeof(bool))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.rng, np * sizeof(RngState))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.pixel_index, np * sizeof(int))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.accum, npx * sizeof(float3))) != cudaSuccess)
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
  cudaFree(b.pixel_index);
  b.pixel_index = nullptr;
}
