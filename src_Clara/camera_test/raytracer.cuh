// raytracer.cuh
// Declaration of the kernel launch wrapper called from main.cpp.

#pragma once
#include <cuda_runtime.h>


// devPtr  — pointer to the mapped PBO (RGBA8, row-major)
// width   — image width in pixels
// height  — image height in pixels
// frame   — frame counter (useful for animations / progressive refinement)
void launchRayTracer(uchar4* devPtr, int width, int height, int frame, cudaStream_t stream);
