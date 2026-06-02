#pragma once

#include <cuda_runtime.h>

#include "camera.cuh"
#include "rng.cuh"
#include "types.cuh"
#include "wavefront.cuh"

// Megakernel (retained for comparison)
cudaError_t launch_init_rng(RngState *d_rng, int image_width, int image_height,
                            uint32_t seed, cudaStream_t stream);

cudaError_t launch_render(uchar3 *d_framebuffer, GpuCamera cam, GpuScene scene,
                          RngState *d_rng, cudaStream_t stream);

// Wavefront pipeline.
// Call order each frame:
//   launch_wavefront_accum_clear (once)
//   for s in 0..samples_per_pixel:
//     launch_wavefront_init       (generates primary ray for this sample)
//     for b in 0..max_depth:
//       launch_wavefront_bounce   (one path bounce)
//   launch_wavefront_finalize     (accum -> RGB)

cudaError_t launch_wavefront_accum_clear(WavefrontBuffers &b,
                                         cudaStream_t stream);

cudaError_t launch_wavefront_init(WavefrontBuffers &b, GpuCamera cam,
                                  uint32_t seed, cudaStream_t stream);

cudaError_t launch_wavefront_bounce(WavefrontBuffers &b, GpuScene scene,
                                    cudaStream_t stream);

cudaError_t launch_wavefront_finalize(const WavefrontBuffers &b, uchar3 *d_out,
                                      int samples_per_pixel,
                                      cudaStream_t stream);
