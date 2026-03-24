#include "cuda_renderer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../cpu/camera.h"
#include "../cpu/hittable_list.h"
#include "../cpu/material.h"
#include "../cpu/sphere.h"
#include "cuda_camera.cuh"
#include "cuda_kernels.cuh"
#include "cuda_types.cuh"

namespace
{

    inline float3 to_float3(const vec3 &v)
    {
        return make_float3(static_cast<float>(v.x()), static_cast<float>(v.y()), static_cast<float>(v.z()));
    }

    bool convert_material(
        const material *mat,
        GpuMaterial &out,
        std::string &error_message)
    {

        if (const auto *lam = dynamic_cast<const lambertian *>(mat))
        {
            out.type = MAT_LAMBERTIAN;
            out.albedo = to_float3(lam->get_albedo());
            out.fuzz = 0.0f;
            out.ref_idx = 1.0f;
            return true;
        }

        if (const auto *met = dynamic_cast<const metal *>(mat))
        {
            out.type = MAT_METAL;
            out.albedo = to_float3(met->get_albedo());
            out.fuzz = static_cast<float>(met->get_fuzz());
            out.ref_idx = 1.0f;
            return true;
        }

        if (const auto *die = dynamic_cast<const dielectric *>(mat))
        {
            out.type = MAT_DIELECTRIC;
            out.albedo = make_float3(1.0f, 1.0f, 1.0f);
            out.fuzz = 0.0f;
            out.ref_idx = static_cast<float>(die->get_refraction_index());
            return true;
        }

        error_message = "Unsupported material type for CUDA MVP conversion.";
        return false;
    }

    bool convert_world(
        const hittable_list &world,
        std::vector<GpuSphere> &spheres,
        std::vector<GpuMaterial> &materials,
        std::string &error_message)
    {

        std::unordered_map<const material *, int> material_map;

        spheres.clear();
        materials.clear();

        for (const auto &object : world.objects)
        {
            const auto *sp = dynamic_cast<const sphere *>(object.get());
            if (!sp)
            {
                error_message = "CUDA MVP currently supports sphere objects only.";
                return false;
            }

            const material *mat_ptr = sp->get_material().get();
            if (!mat_ptr)
            {
                error_message = "Sphere has null material pointer.";
                return false;
            }

            int mat_index = -1;
            auto it = material_map.find(mat_ptr);
            if (it == material_map.end())
            {
                GpuMaterial gpu_mat{};
                if (!convert_material(mat_ptr, gpu_mat, error_message))
                {
                    return false;
                }
                mat_index = static_cast<int>(materials.size());
                materials.push_back(gpu_mat);
                material_map.insert({mat_ptr, mat_index});
            }
            else
            {
                mat_index = it->second;
            }

            GpuSphere gpu_sphere{};
            gpu_sphere.center = to_float3(sp->get_center());
            gpu_sphere.radius = static_cast<float>(sp->get_radius());
            gpu_sphere.material_index = mat_index;
            spheres.push_back(gpu_sphere);
        }

        return true;
    }

    GpuCamera build_gpu_camera(const camera &cam)
    {
        GpuCamera out{};

        int image_height = static_cast<int>(cam.image_width / cam.aspect_ratio);
        if (image_height < 1)
        {
            image_height = 1;
        }

        double theta = degrees_to_radians(cam.vfov);
        double h = std::tan(theta * 0.5);
        double viewport_height = 2.0 * h * cam.focus_dist;
        double viewport_width = viewport_height * (static_cast<double>(cam.image_width) / image_height);

        vec3 w = unit_vector(cam.look_from - cam.look_at);
        vec3 u = unit_vector(cross(cam.vup, w));
        vec3 v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        vec3 pixel_delta_u = viewport_u / cam.image_width;
        vec3 pixel_delta_v = viewport_v / image_height;

        point3 center = cam.look_from;
        point3 viewport_upper_left = center - (cam.focus_dist * w) - viewport_u / 2.0 - viewport_v / 2.0;
        point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        double defocus_radius = cam.focus_dist * std::tan(degrees_to_radians(cam.defocus_angle * 0.5));

        out.image_width = cam.image_width;
        out.image_height = image_height;
        out.samples_per_pixel = cam.samples_per_pixel;
        out.max_depth = cam.max_depth;

        out.center = to_float3(center);
        out.pixel00_loc = to_float3(pixel00_loc);
        out.pixel_delta_u = to_float3(pixel_delta_u);
        out.pixel_delta_v = to_float3(pixel_delta_v);

        out.defocus_disk_u = to_float3(u * defocus_radius);
        out.defocus_disk_v = to_float3(v * defocus_radius);
        out.defocus_angle = static_cast<float>(cam.defocus_angle);

        return out;
    }

    bool write_ppm(const std::string &path, int width, int height, const std::vector<uchar3> &pixels)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open())
        {
            return false;
        }

        out << "P3\n"
            << width << " " << height << "\n255\n";
        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                const uchar3 &c = pixels[j * width + i];
                out << static_cast<int>(c.x) << ' '
                    << static_cast<int>(c.y) << ' '
                    << static_cast<int>(c.z) << '\n';
            }
        }

        return true;
    }

} // namespace

bool render_cuda_mvp(
    const camera &cam,
    const hittable_list &world,
    const std::string &output_path,
    double &elapsed_seconds,
    std::string &error_message)
{

    elapsed_seconds = 0.0;

    std::vector<GpuSphere> h_spheres;
    std::vector<GpuMaterial> h_materials;

    if (!convert_world(world, h_spheres, h_materials, error_message))
    {
        return false;
    }

    if (h_spheres.empty())
    {
        error_message = "Scene is empty; nothing to render.";
        return false;
    }

    GpuCamera gpu_cam = build_gpu_camera(cam);
    int pixel_count = gpu_cam.image_width * gpu_cam.image_height;

    GpuSphere *d_spheres = nullptr;
    GpuMaterial *d_materials = nullptr;
    RngState *d_rng = nullptr;
    uchar3 *d_framebuffer = nullptr;

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
    };

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void **>(&d_spheres), h_spheres.size() * sizeof(GpuSphere));
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_materials), h_materials.size() * sizeof(GpuMaterial));
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

    err = cudaMemcpy(d_spheres, h_spheres.data(), h_spheres.size() * sizeof(GpuSphere), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    err = cudaMemcpy(d_materials, h_materials.data(), h_materials.size() * sizeof(GpuMaterial), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        error_message = cudaGetErrorString(err);
        cleanup();
        return false;
    }

    GpuScene scene{};
    scene.spheres = d_spheres;
    scene.sphere_count = static_cast<int>(h_spheres.size());
    scene.materials = d_materials;
    scene.material_count = static_cast<int>(h_materials.size());

    err = launch_init_rng(d_rng, gpu_cam.image_width, gpu_cam.image_height, 1337u, nullptr);
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
    err = launch_render(d_framebuffer, gpu_cam, scene, d_rng, nullptr);
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

    if (!write_ppm(output_path, gpu_cam.image_width, gpu_cam.image_height, h_framebuffer))
    {
        error_message = "Failed to write CUDA output image file.";
        cleanup();
        return false;
    }

    cleanup();
    return true;
}
