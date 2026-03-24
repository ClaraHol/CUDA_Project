# CUDA Migration Draft (CPU -> OpenMP -> CUDA)

## Goal
Keep three comparable renderer paths:
- Sequential CPU baseline
- OpenMP CPU parallel baseline
- CUDA GPU implementation

The objective is to preserve image behavior while reducing render time on the GPU.

## Current CPU Hot Path
Your expensive loop is currently in `camera::render_parallel` and `camera::render`:
- Pixel loop: `for j`, `for i`
- Monte Carlo loop: `for sample`
- Recursive shading: `ray_color(..., depth, world)`

This is a good fit for CUDA where each thread computes one pixel (or one sample-per-pixel strategy).

## High-Level Migration Strategy
1. Keep the CPU renderer untouched for correctness and timing baselines.
2. Add a separate CUDA renderer path in new files under `src/cuda`.
3. Flatten scene data (spheres/materials) into contiguous arrays for GPU.
4. Port camera/ray/math helpers to device-safe functions.
5. Replace recursive ray tracing with iterative bounce loop on GPU.
6. Add CUDA timing and compare against CPU/OpenMP.

## Why Not Directly Reuse Existing World Types
Current scene uses polymorphism and `shared_ptr`:
- `hittable` virtual `hit`
- `material` virtual `scatter`
- `hittable_list` with `vector<shared_ptr<hittable>>`

This is good C++ design on CPU but not ideal for CUDA performance and portability. A struct-of-arrays or array-of-structs with enum-based material dispatch is preferred.

## Proposed Folder Layout
Create these files:

- `src/cuda/cuda_types.cuh`
- `src/cuda/cuda_camera.cuh`
- `src/cuda/cuda_rng.cuh`
- `src/cuda/cuda_kernels.cu`
- `src/cuda/cuda_renderer.h`
- `src/cuda/cuda_renderer.cu`

Keep existing CPU files as-is.

## Data Model Draft (GPU Side)
Use simple POD structs:

```cpp
// cuda_types.cuh
#pragma once

enum MaterialType : int {
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_DIELECTRIC = 2,
};

struct GpuMaterial {
    MaterialType type;
    float3 albedo;
    float fuzz;
    float ref_idx;
};

struct GpuSphere {
    float3 center;
    float radius;
    int material_index;
};

struct GpuScene {
    const GpuSphere* spheres;
    int sphere_count;
    const GpuMaterial* materials;
    int material_count;
};

struct GpuCamera {
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;

    float3 center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    float3 defocus_disk_u;
    float3 defocus_disk_v;
    float defocus_angle;
};
```

## Kernel Execution Draft
Thread mapping:
- 2D grid over image dimensions
- One thread computes one pixel

Pseudo-flow:

```cpp
__global__ void render_kernel(
    uchar3* out_rgb,
    GpuCamera cam,
    GpuScene scene,
    RngState* rng_states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cam.image_width || y >= cam.image_height) return;

    int pixel_idx = y * cam.image_width + x;
    RngState rng = rng_states[pixel_idx];

    float3 accum = make_float3(0,0,0);
    for (int s = 0; s < cam.samples_per_pixel; ++s) {
        Ray r = get_ray_device(cam, x, y, rng);
        accum += ray_color_iterative(r, cam.max_depth, scene, rng);
    }

    rng_states[pixel_idx] = rng;
    out_rgb[pixel_idx] = to_rgb_gamma_corrected(accum / cam.samples_per_pixel);
}
```

## Critical Porting Changes
1. Replace recursion in `ray_color` with a loop:
- Track `throughput` (attenuation product)
- For each bounce:
- Find closest hit
- Scatter/update ray and throughput
- Break on miss or absorption

2. Replace random generator:
- CPU RNG in `rt_weekend.h` relies on OpenMP thread local state.
- GPU needs per-thread RNG state (`curandStatePhilox4_32_10_t` or custom XOR state).

3. Replace virtual dispatch:
- material scatter done with `switch(material.type)`.

## Host-Side Integration Draft
Add a CUDA render entry point from `main.cpp`:

```cpp
// cpu code still creates world as today
// convert world -> flat GPU arrays

CudaRenderer renderer;
renderer.render(world, cam, "image_cuda.ppm");
```

Inside renderer:
1. Build host vectors of `GpuMaterial` and `GpuSphere`
2. `cudaMalloc` + `cudaMemcpy` to device
3. Launch setup RNG kernel
4. Launch render kernel
5. Copy `uchar3` framebuffer back
6. Save PPM
7. Free device memory

## Suggested Incremental Milestones
1. MVP kernel:
- Spheres only
- Lambertian only
- No defocus blur
- Small image and low spp

2. Feature parity:
- Add metal + dielectric
- Add defocus camera
- Match CPU visual output statistically

3. Performance pass:
- Tune block size (`8x8`, `16x16`, `32x8`)
- Reduce branch divergence in material scatter
- Prefer `float` math unless precision requires double

4. Acceleration structure:
- Add BVH when sphere count grows

## Build Draft (Windows, NVCC)
Option A: Add a dedicated CUDA build target while keeping current CPU Makefile untouched.

Example command shape:

```powershell
nvcc -O3 -std=c++17 -Xcompiler="/openmp" \
  src/cuda/cuda_renderer.cu src/cuda/cuda_kernels.cu src/cpu/main.cpp \
  -o build/raytrace_cuda.exe
```

If you keep MinGW for CPU and MSVC toolchain for CUDA, verify ABI/toolchain compatibility early.

## Benchmarking Rules
For fair comparisons:
1. Use identical scene/camera/spp/depth.
2. Measure render-only time separately from image output.
3. Warm up CUDA once before timed runs.
4. Use CUDA events for kernel timing and synchronized end-to-end timing.

## Validation Checklist
- CPU sequential image exists and timing recorded
- CPU OpenMP image exists and timing recorded
- CUDA image visually comparable
- CUDA runtime significantly lower at moderate/high resolution and spp
- No obvious fireflies/artifact regressions beyond expected Monte Carlo noise

## Concrete Next Implementation Step
Start with this first coding slice:
1. Add `src/cuda/cuda_types.cuh` with GPU scene/camera structs.
2. Add `src/cuda/cuda_rng.cuh` with per-pixel RNG.
3. Add `src/cuda/cuda_kernels.cu` with:
- `init_rng_kernel`
- `render_kernel` (lambertian-only MVP)
4. Add `src/cuda/cuda_renderer.cu/.h` with host memory management and PPM output.
5. Add a `main` flag to choose mode: `--mode cpu|omp|cuda`.

---
This draft is intentionally implementation-oriented so you can move in small verified steps without breaking your current baseline pipeline.
