#pragma once

#include <cuda_runtime.h>

#include "camera.cuh"
#include "rng.cuh"
#include "types.cuh"
#include "wavefront.cuh"

cudaError_t launch_init_rng(RngState *d_rng, int image_width, int image_height,
                            uint32_t seed, cudaStream_t stream);

cudaError_t launch_render(uchar3 *d_framebuffer, GpuCamera cam, GpuScene scene,
                          RngState *d_rng, cudaStream_t stream);

// Wavefront kernels
cudaError_t launch_wavefront_init(WavefrontBuffers &b, GpuCamera cam,
                                  cudaStream_t stream);

cudaError_t launch_wavefront_bounce(WavefrontBuffers &b, GpuScene scene,
                                    cudaStream_t stream);

cudaError_t launch_wavefront_finalize(const WavefrontBuffers &b, uchar3 *d_out,
                                      GpuCamera cam, cudaStream_t stream);