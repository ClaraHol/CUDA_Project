#include "cuda_kernels.cuh"

#include <math.h>

namespace
{

    inline __device__ float3 random_in_unit_sphere(RngState &rng)
    {
        while (true)
        {
            float3 p = make_vec3(
                2.0f * rng_next_f32(rng) - 1.0f,
                2.0f * rng_next_f32(rng) - 1.0f,
                2.0f * rng_next_f32(rng) - 1.0f);
            float lsq = len_sq3(p);
            if (lsq > 1e-8f && lsq <= 1.0f)
            {
                return p;
            }
        }
    }

    inline __device__ float3 random_unit_vector(RngState &rng)
    {
        return unit3(random_in_unit_sphere(rng));
    }

    inline __device__ float3 random_in_unit_disk(RngState &rng)
    {
        while (true)
        {
            float3 p = make_vec3(
                2.0f * rng_next_f32(rng) - 1.0f,
                2.0f * rng_next_f32(rng) - 1.0f,
                0.0f);
            if (len_sq3(p) < 1.0f)
            {
                return p;
            }
        }
    }

    inline __device__ bool hit_sphere(const GpuSphere &s, const Ray &r, float t_min, float t_max, Hit &out_hit)
    {
        float3 oc = sub3(s.center, r.origin);
        float a = len_sq3(r.dir);
        float h = dot3(r.dir, oc);
        float c = len_sq3(oc) - s.radius * s.radius;
        float disc = h * h - a * c;

        if (disc < 0.0f)
        {
            return false;
        }

        float sqrtd = sqrtf(disc);

        float root = (h - sqrtd) / a;
        if (root <= t_min || root >= t_max)
        {
            root = (h + sqrtd) / a;
            if (root <= t_min || root >= t_max)
            {
                return false;
            }
        }

        out_hit.t = root;
        out_hit.p = ray_at(r, root);
        float3 outward_normal = div3(sub3(out_hit.p, s.center), s.radius);
        set_face_normal(out_hit, r, outward_normal);
        out_hit.material_index = s.material_index;

        return true;
    }

    inline __device__ bool hit_scene(const GpuScene &scene, const Ray &r, float t_min, float t_max, Hit &rec)
    {
        Hit tmp;
        bool hit_anything = false;
        float closest = t_max;

        for (int i = 0; i < scene.sphere_count; ++i)
        {
            if (hit_sphere(scene.spheres[i], r, t_min, closest, tmp))
            {
                hit_anything = true;
                closest = tmp.t;
                rec = tmp;
            }
        }

        return hit_anything;
    }

    inline __device__ float reflectance(float cosine, float ref_idx)
    {
        float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
    }

    inline __device__ bool scatter(
        const GpuMaterial &mat,
        const Ray &r_in,
        const Hit &hit,
        RngState &rng,
        float3 &attenuation,
        Ray &scattered)
    {

        if (mat.type == MAT_LAMBERTIAN)
        {
            float3 scatter_dir = add3(hit.normal, random_unit_vector(rng));
            if (len_sq3(scatter_dir) < 1e-8f)
            {
                scatter_dir = hit.normal;
            }
            scattered.origin = hit.p;
            scattered.dir = scatter_dir;
            attenuation = mat.albedo;
            return true;
        }

        if (mat.type == MAT_METAL)
        {
            float3 reflected = reflect3(unit3(r_in.dir), hit.normal);
            reflected = add3(reflected, mul3(random_unit_vector(rng), mat.fuzz));
            scattered.origin = hit.p;
            scattered.dir = reflected;
            attenuation = mat.albedo;
            return dot3(scattered.dir, hit.normal) > 0.0f;
        }

        if (mat.type == MAT_DIELECTRIC)
        {
            attenuation = make_vec3(1.0f, 1.0f, 1.0f);
            float eta_ratio = hit.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;

            float3 unit_dir = unit3(r_in.dir);
            float cos_theta = fminf(dot3(mul3(unit_dir, -1.0f), hit.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = eta_ratio * sin_theta > 1.0f;
            float3 direction;
            if (cannot_refract || reflectance(cos_theta, eta_ratio) > rng_next_f32(rng))
            {
                direction = reflect3(unit_dir, hit.normal);
            }
            else
            {
                direction = refract3(unit_dir, hit.normal, eta_ratio);
            }

            scattered.origin = hit.p;
            scattered.dir = direction;
            return true;
        }

        return false;
    }

    inline __device__ float3 sky_color(const Ray &r)
    {
        float3 unit_dir = unit3(r.dir);
        float a = 0.5f * (unit_dir.y + 1.0f);
        float3 c0 = make_vec3(1.0f, 1.0f, 1.0f);
        float3 c1 = make_vec3(0.5f, 0.7f, 1.0f);
        return add3(mul3(c0, 1.0f - a), mul3(c1, a));
    }

    inline __device__ Ray get_ray(const GpuCamera &cam, int px, int py, RngState &rng)
    {
        float ox = rng_next_f32(rng) - 0.5f;
        float oy = rng_next_f32(rng) - 0.5f;

        float3 pixel_sample = add3(
            cam.pixel00_loc,
            add3(
                mul3(cam.pixel_delta_u, static_cast<float>(px) + ox),
                mul3(cam.pixel_delta_v, static_cast<float>(py) + oy)));

        float3 origin = cam.center;
        if (cam.defocus_angle > 0.0f)
        {
            float3 p = random_in_unit_disk(rng);
            origin = add3(cam.center, add3(mul3(cam.defocus_disk_u, p.x), mul3(cam.defocus_disk_v, p.y)));
        }

        Ray r;
        r.origin = origin;
        r.dir = sub3(pixel_sample, origin);
        return r;
    }

    inline __device__ float3 ray_color_iterative(const Ray &initial_ray, int max_depth, const GpuScene &scene, RngState &rng)
    {
        Ray ray = initial_ray;
        float3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
        float3 radiance = make_vec3(0.0f, 0.0f, 0.0f);

        for (int bounce = 0; bounce < max_depth; ++bounce)
        {
            Hit rec;
            if (!hit_scene(scene, ray, 0.001f, 1e30f, rec))
            {
                radiance = add3(radiance, mul3(throughput, sky_color(ray)));
                break;
            }

            const GpuMaterial &mat = scene.materials[rec.material_index];
            float3 attenuation;
            Ray scattered;
            if (!scatter(mat, ray, rec, rng, attenuation, scattered))
            {
                break;
            }

            throughput = mul3(throughput, attenuation);
            ray = scattered;
        }

        return radiance;
    }

    inline __device__ uchar3 to_rgb8(const float3 &c)
    {
        float r = sqrtf(fmaxf(c.x, 0.0f));
        float g = sqrtf(fmaxf(c.y, 0.0f));
        float b = sqrtf(fmaxf(c.z, 0.0f));

        unsigned char rb = static_cast<unsigned char>(256.0f * clampf(r, 0.0f, 0.999f));
        unsigned char gb = static_cast<unsigned char>(256.0f * clampf(g, 0.0f, 0.999f));
        unsigned char bb = static_cast<unsigned char>(256.0f * clampf(b, 0.0f, 0.999f));

        return make_uchar3(rb, gb, bb);
    }

} // namespace

__global__ void init_rng_kernel(RngState *rng, int width, int height, uint32_t seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int idx = y * width + x;
    rng_seed(rng[idx], seed, static_cast<uint32_t>(idx));
}

__global__ void render_kernel(uchar3 *out_rgb, GpuCamera cam, GpuScene scene, RngState *rng_states)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cam.image_width || y >= cam.image_height)
    {
        return;
    }

    int idx = y * cam.image_width + x;
    RngState rng = rng_states[idx];

    float3 accum = make_vec3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < cam.samples_per_pixel; ++s)
    {
        Ray ray = get_ray(cam, x, y, rng);
        accum = add3(accum, ray_color_iterative(ray, cam.max_depth, scene, rng));
    }

    float inv_spp = 1.0f / static_cast<float>(cam.samples_per_pixel);
    out_rgb[idx] = to_rgb8(mul3(accum, inv_spp));
    rng_states[idx] = rng;
}

cudaError_t launch_init_rng(
    RngState *d_rng,
    int image_width,
    int image_height,
    uint32_t seed,
    cudaStream_t stream)
{

    dim3 block(16, 16);
    dim3 grid(
        (image_width + block.x - 1) / block.x,
        (image_height + block.y - 1) / block.y);

    init_rng_kernel<<<grid, block, 0, stream>>>(d_rng, image_width, image_height, seed);
    return cudaGetLastError();
}

cudaError_t launch_render(
    uchar3 *d_framebuffer,
    GpuCamera cam,
    GpuScene scene,
    RngState *d_rng,
    cudaStream_t stream)
{

    dim3 block(16, 16);
    dim3 grid(
        (cam.image_width + block.x - 1) / block.x,
        (cam.image_height + block.y - 1) / block.y);

    render_kernel<<<grid, block, 0, stream>>>(d_framebuffer, cam, scene, d_rng);
    return cudaGetLastError();
}
