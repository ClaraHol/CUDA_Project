#pragma once

#include "rng.cuh"
#include "types.cuh"
#include <cuda_runtime.h>

// One slot per path (total_paths = width * height * samples_per_pixel).
// Paths are laid out as:
//   [px0_s0, px0_s1, ..., px0_s(SPP-1), px1_s0, ...]
// so pixel_index[i] == i / samples_per_pixel, but we store it explicitly
// to keep the finalize kernel independent of SPP.
struct WavefrontBuffers {
  Ray *rays;          // current ray for each path
  float3 *throughput; // accumulated path throughput (starts at (1,1,1))
  int *pixel_index;   // which pixel this path contributes to
  bool *active;       // false once the path misses the scene or is terminated
  float3 *accum;      // per-pixel radiance accumulator (atomicAdd target)
  RngState *rng;      // per-path RNG state

  int total_paths;  // width * height * samples_per_pixel
  int total_pixels; // width * height
  int samples_per_pixel;
};

inline cudaError_t allocate_wavefront_buffers(WavefrontBuffers &b, int width,
                                              int height,
                                              int samples_per_pixel) {
  b.samples_per_pixel = samples_per_pixel;
  b.total_paths = width * height * samples_per_pixel;
  b.total_pixels = width * height;

  auto alloc = [&](auto **ptr, size_t bytes) -> cudaError_t {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess)
      *ptr = nullptr;
    return err;
  };

  cudaError_t err;
  if ((err = alloc(&b.rays, b.total_paths * sizeof(Ray))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.throughput, b.total_paths * sizeof(float3))) !=
      cudaSuccess)
    return err;
  if ((err = alloc(&b.pixel_index, b.total_paths * sizeof(int))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.active, b.total_paths * sizeof(bool))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.accum, b.total_pixels * sizeof(float3))) != cudaSuccess)
    return err;
  if ((err = alloc(&b.rng, b.total_paths * sizeof(RngState))) != cudaSuccess)
    return err;

  return cudaSuccess;
}

inline void free_wavefront_buffers(WavefrontBuffers &b) {
  cudaFree(b.rays);
  b.rays = nullptr;
  cudaFree(b.throughput);
  b.throughput = nullptr;
  cudaFree(b.pixel_index);
  b.pixel_index = nullptr;
  cudaFree(b.active);
  b.active = nullptr;
  cudaFree(b.accum);
  b.accum = nullptr;
  cudaFree(b.rng);
  b.rng = nullptr;
}
