// raytracer.cu
// Minimal kernel stub — replace the body of rayTracerKernel with your
// actual ray tracing code.

#include "raytracer.cuh"
#include <cuda_runtime.h>
#include <cstdio>
// ---------------------------------------------------------------------------
// Kernel: one thread per pixel
// ---------------------------------------------------------------------------

// Forward declare the kernel so launchRayTracer can call it
__global__ void rayTracerKernel(uchar4* out, int width, int height, int frame);

__global__ void rayTracerKernel(uchar4* out, int width, int height, int frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // --- Replace everything below with your ray tracing logic ---

    // Normalised UV coords (0..1)
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    // Animated gradient placeholder so you can verify the pipeline works
    float t = (float)frame * 0.01f;
    unsigned char r = (unsigned char)((u + 0.5f * sinf(t))         * 255.f);
    unsigned char g = (unsigned char)((v + 0.5f * cosf(t * 0.7f))  * 255.f);
    unsigned char b = (unsigned char)(0.5f + 0.5f * sinf(t * 1.3f) * 255.f);

    out[idx] = make_uchar4(r, g, b, 255);
}


// ---------------------------------------------------------------------------
// Launch wrapper (called from main.cpp)
// ---------------------------------------------------------------------------
// raytracer.cu

void launchRayTracer(uchar4* devPtr, int width, int height,
                     int frame, cudaStream_t stream)
{
    dim3 blockSize(8, 8);
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    rayTracerKernel<<<gridSize, blockSize, 0, stream>>>(devPtr, width, height, frame);
}
    

