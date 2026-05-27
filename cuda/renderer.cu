#include "renderer.cuh"

#include <cuda_runtime.h>

#include "kernels.cuh"
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "bvh_builder.cuh"
#include "stb_image_write.h"
#include "wavefront.cuh"

namespace {

__host__ bool write_color_png(const uchar3 *framebuffer, int num_pixels,
                              int image_width, int image_height,
                              const std::string &output_name) {
  std::vector<uint8_t> pixels(num_pixels * 3);
  for (int i = 0; i < num_pixels; i++) {
    uchar3 c = framebuffer[i];
    pixels[i * 3 + 0] = c.x;
    pixels[i * 3 + 1] = c.y;
    pixels[i * 3 + 2] = c.z;
  }
  return stbi_write_png(output_name.c_str(), image_width, image_height, 3,
                        pixels.data(), image_width * 3) == 1;
}

} // namespace

bool render_cuda_scene(const GpuCamera &cam,
                       const std::vector<GpuSphere> &spheres,
                       const std::vector<GpuMaterial> &materials,
                       const std::string &output_path, double &elapsed_seconds,
                       std::string &error_message) {
  elapsed_seconds = 0.0;

  if (spheres.empty()) {
    error_message = "Scene is empty.";
    return false;
  }
  if (materials.empty()) {
    error_message = "Scene has no materials.";
    return false;
  }

  // -------------------------------------------------------------------------
  // Build BVH on the host
  // -------------------------------------------------------------------------
  BVHBuildResult bvh = build_bvh_for_spheres(spheres, 4);
  if (bvh.nodes.empty()) {
    error_message = "BVH build failed.";
    return false;
  }

  const int pixel_count = cam.image_width * cam.image_height;

  // -------------------------------------------------------------------------
  // Device allocations
  // -------------------------------------------------------------------------
  GpuSphere *d_spheres = nullptr;
  GpuMaterial *d_materials = nullptr;
  uchar3 *d_framebuffer = nullptr;
  BVHNode *d_bvh_nodes = nullptr;
  uint32_t *d_bvh_prim = nullptr;

  std::vector<uchar3> h_framebuffer(static_cast<size_t>(pixel_count));

  auto cleanup = [&]() {
    cudaFree(d_framebuffer);
    cudaFree(d_materials);
    cudaFree(d_spheres);
    cudaFree(d_bvh_prim);
    cudaFree(d_bvh_nodes);
  };

  cudaError_t err = cudaSuccess;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    err = (call);                                                              \
    if (err != cudaSuccess) {                                                  \
      error_message = cudaGetErrorString(err);                                 \
      cleanup();                                                               \
      return false;                                                            \
    }                                                                          \
  } while (0)

  CUDA_CHECK(cudaMalloc(&d_spheres, spheres.size() * sizeof(GpuSphere)));
  CUDA_CHECK(cudaMalloc(&d_materials, materials.size() * sizeof(GpuMaterial)));
  CUDA_CHECK(cudaMalloc(&d_framebuffer, pixel_count * sizeof(uchar3)));
  CUDA_CHECK(cudaMalloc(&d_bvh_nodes, bvh.nodes.size() * sizeof(BVHNode)));
  CUDA_CHECK(
      cudaMalloc(&d_bvh_prim, bvh.primitive_indices.size() * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemcpy(d_spheres, spheres.data(),
                        spheres.size() * sizeof(GpuSphere),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_materials, materials.data(),
                        materials.size() * sizeof(GpuMaterial),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bvh_nodes, bvh.nodes.data(),
                        bvh.nodes.size() * sizeof(BVHNode),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bvh_prim, bvh.primitive_indices.data(),
                        bvh.primitive_indices.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  GpuScene scene{};
  scene.spheres = d_spheres;
  scene.sphere_count = static_cast<int>(spheres.size());
  scene.materials = d_materials;
  scene.material_count = static_cast<int>(materials.size());
  scene.bvh_nodes = d_bvh_nodes;
  scene.bvh_node_count = static_cast<int>(bvh.nodes.size());
  scene.bvh_primitive_indices = d_bvh_prim;

  // -------------------------------------------------------------------------
  // Wavefront buffers (sized to total_pixels, no SPP multiplier)
  // -------------------------------------------------------------------------
  WavefrontBuffers wf{};
  err = allocate_wavefront_buffers(wf, cam.image_width, cam.image_height);
  if (err != cudaSuccess) {
    error_message = cudaGetErrorString(err);
    free_wavefront_buffers(wf);
    cleanup();
    return false;
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  cudaEvent_t ev_start{}, ev_stop{};
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start);

  // Zero accumulator once — it will be written by all sample launches.
  CUDA_CHECK(launch_wavefront_accum_clear(wf, nullptr));

  constexpr uint32_t BASE_SEED = 0xA7B3C1D5u;

  for (int s = 0; s < cam.samples_per_pixel; ++s) {
    // Generate primary rays for this sample. Each (pixel, sample) pair
    // gets a statistically independent RNG lane via the sample index.
    CUDA_CHECK(launch_wavefront_init(wf, cam, BASE_SEED, s, nullptr));

    // Bounce up to max_depth times. Inactive paths are no-ops.
    for (int b = 0; b < cam.max_depth; ++b) {
      CUDA_CHECK(launch_wavefront_bounce(wf, scene, nullptr));
    }
  }

  // Divide accum by SPP and gamma-encode to the output framebuffer.
  CUDA_CHECK(launch_wavefront_finalize(wf, d_framebuffer, cam.samples_per_pixel,
                                       nullptr));

  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  elapsed_seconds = static_cast<double>(ms) / 1000.0;

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  // -------------------------------------------------------------------------
  // Readback and write PNG
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(h_framebuffer.data(), d_framebuffer,
                        pixel_count * sizeof(uchar3), cudaMemcpyDeviceToHost));

  free_wavefront_buffers(wf);
  cleanup();

  if (!write_color_png(h_framebuffer.data(), pixel_count, cam.image_width,
                       cam.image_height, output_path)) {
    error_message = "Failed to write output PNG.";
    return false;
  }

#undef CUDA_CHECK
  return true;
}
