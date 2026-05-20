#include "renderer.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <string>
#include <vector>
#include "kernels.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "bvh_builder.cuh"

namespace
{

    __host__ bool write_color_png(const uchar3 *framebuffer, int num_pixels, int image_width, int image_height, const std::string &output_name)
    {
        std::vector<uint8_t> pixels(num_pixels * 3);
        for (int i = 0; i < num_pixels; i++)
        {
            uchar3 c = framebuffer[i];
            pixels[i * 3 + 0] = c.x;
            pixels[i * 3 + 1] = c.y;
            pixels[i * 3 + 2] = c.z;
        }
        return stbi_write_png(output_name.c_str(), image_width, image_height, 3, pixels.data(), image_width * 3) == 1;
    }

} // namespace

bool render_cuda_scene(
    const GpuCamera &cam,
    const std::vector<GpuSphere> &spheres,
    const std::vector<GpuMaterial> &materials,
    const std::string &output_path,
    double &elapsed_seconds,
    std::string &error_message)
{

    elapsed_seconds = 0.0;

    if (spheres.empty())
    {
        error_message = "Scene is empty; nothing to render.";
        return false;
    }
    if (materials.empty())
    {
        error_message = "Scene has no materials.";
        return false;
    }

    // Build BVH
    BVHBuildResult bvh = build_bvh_for_spheres(spheres, 4);
    if (bvh.nodes.empty())
    {
        error_message = "BVH build failed.";
        return false;
    }

    int pixel_count = cam.image_width * cam.image_height;

    GpuSphere *d_spheres = nullptr;
    GpuMaterial *d_materials = nullptr;
    RngState *d_rng = nullptr;
    uchar3 *d_framebuffer = nullptr;
    // BVH
    BVHNode *d_bvh_nodes = nullptr;
    uint32_t *d_bvh_primitive_indices = nullptr;

    std::vector<uchar3> h_framebuffer(static_cast<size_t>(pixel_count));

    auto cleanup = [&]()
    {
        if (d_framebuffer)
            cudaFree(d_framebuffer);
        if (d_rng)
            cudaFree(d_rng);
        if (d_materials)
            cudaFree(d_materials);
        if (d_spheres)
            cudaFree(d_spheres);
        if (d_bvh_primitive_indices)
            cudaFree(d_bvh_primitive_indices);
        if (d_bvh_nodes)
            cudaFree(d_bvh_nodes);
    };

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void **>(&d_spheres), spheres.size() * sizeof(GpuSphere));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_materials), materials.size() * sizeof(GpuMaterial));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_rng), pixel_count * sizeof(RngState));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_framebuffer), pixel_count * sizeof(uchar3));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(d_spheres, spheres.data(), spheres.size() * sizeof(GpuSphere), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(d_materials, materials.data(), materials.size() * sizeof(GpuMaterial), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    // BVH - Nodes
    err = cudaMalloc(reinterpret_cast<void **>(&d_bvh_nodes), bvh.nodes.size() * sizeof(BVHNode));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(d_bvh_nodes, bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    // BVH - Primitive indices
    err = cudaMalloc(reinterpret_cast<void **>(&d_bvh_primitive_indices), bvh.primitive_indices.size() * sizeof(uint32_t));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(d_bvh_primitive_indices, bvh.primitive_indices.data(), bvh.primitive_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    GpuScene scene{};
    scene.spheres = d_spheres;
    scene.sphere_count = static_cast<int>(spheres.size());
    scene.materials = d_materials;
    scene.material_count = static_cast<int>(materials.size());
    // BVH
    scene.bvh_nodes = d_bvh_nodes;
    scene.bvh_node_count = static_cast<int>(bvh.nodes.size());
    scene.bvh_primitive_indices = d_bvh_primitive_indices;

    err = launch_init_rng(d_rng, cam.image_width, cam.image_height, 1337u, nullptr);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    err = launch_render(d_framebuffer, cam, scene, d_rng, nullptr);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cleanup();
        return false;
    }

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event, stop_event);
    elapsed_seconds = static_cast<double>(ms) / 1000.0;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    err = cudaMemcpy(h_framebuffer.data(), d_framebuffer, pixel_count * sizeof(uchar3), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    if (!write_color_png(h_framebuffer.data(), pixel_count, cam.image_width, cam.image_height, output_path))
    {
        error_message = "Failed to write CUDA output PNG file.";
        cleanup();
        return false;
    }

    cleanup();
    return true;
}
