#pragma once

#include <cuda_runtime.h>

#include "cuda_camera.cuh"
#include "cuda_rng.cuh"
#include "cuda_types.cuh"

cudaError_t launch_init_rng(
    RngState *d_rng,
    int image_width,
    int image_height,
    uint32_t seed,
    cudaStream_t stream);

cudaError_t launch_render(
    uchar3 *d_framebuffer,
    GpuCamera cam,
    GpuScene scene,
    RngState *d_rng,
    cudaStream_t stream);
