#include "renderer.cuh"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    float deg_to_rad(float deg)
    {
        return deg * 3.14159265358979323846f / 180.0f;
    }

    float randf(std::mt19937 &rng)
    {
        static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }

    float randf(std::mt19937 &rng, float lo, float hi)
    {
        return lo + (hi - lo) * randf(rng);
    }

    float3 random_color(std::mt19937 &rng)
    {
        return make_vec3(randf(rng), randf(rng), randf(rng));
    }

    float3 random_color(std::mt19937 &rng, float lo, float hi)
    {
        return make_vec3(randf(rng, lo, hi), randf(rng, lo, hi), randf(rng, lo, hi));
    }

    std::string get_output_filename(const std::string &scene_name)
    {
        return std::string("images/") + scene_name + "_cuda.png";
    }

    GpuMaterial make_lambertian(const float3 &albedo)
    {
        GpuMaterial m{};
        m.type = MAT_LAMBERTIAN;
        m.albedo = albedo;
        m.fuzz = 0.0f;
        m.ref_idx = 1.0f;
        return m;
    }

    // GpuMaterial make_metal(const float3 &albedo, float fuzz)
    // {
    //     GpuMaterial m{};
    //     m.type = MAT_METAL;
    //     m.albedo = albedo;
    //     m.fuzz = fuzz;
    //     m.ref_idx = 1.0f;
    //     return m;
    // }

    // GpuMaterial make_dielectric(float ref_idx)
    // {
    //     GpuMaterial m{};
    //     m.type = MAT_DIELECTRIC;
    //     m.albedo = make_vec3(1.0f, 1.0f, 1.0f);
    //     m.fuzz = 0.0f;
    //     m.ref_idx = ref_idx;
    //     return m;
    // }

    void add_sphere(std::vector<GpuSphere> &spheres, const float3 &center, float radius, int mat_index)
    {
        GpuSphere s{};
        s.center = center;
        s.radius = radius;
        s.material_index = mat_index;
        spheres.push_back(s);
    }

    GpuCamera build_camera(
        int image_width,
        float aspect_ratio,
        int samples,
        int max_depth,
        float vfov,
        const float3 &look_from,
        const float3 &look_at,
        const float3 &vup,
        float defocus_angle,
        float focus_dist)
    {
        GpuCamera cam{};

        int image_height = static_cast<int>(image_width / aspect_ratio);
        if (image_height < 1)
        {
            image_height = 1;
        }

        float theta = deg_to_rad(vfov);
        float h = std::tan(theta * 0.5f);
        float viewport_height = 2.0f * h * focus_dist;
        float viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);

        float3 w = unit3(sub3(look_from, look_at));
        float3 u = unit3(cross3(vup, w));
        float3 v = cross3(w, u);

        float3 viewport_u = mul3(u, viewport_width);
        float3 viewport_v = mul3(v, -viewport_height);

        float3 pixel_delta_u = div3(viewport_u, static_cast<float>(image_width));
        float3 pixel_delta_v = div3(viewport_v, static_cast<float>(image_height));

        float3 viewport_upper_left = sub3(sub3(sub3(look_from, mul3(w, focus_dist)), div3(viewport_u, 2.0f)), div3(viewport_v, 2.0f));
        float3 pixel00_loc = add3(viewport_upper_left, mul3(add3(pixel_delta_u, pixel_delta_v), 0.5f));

        float defocus_radius = focus_dist * std::tan(deg_to_rad(defocus_angle * 0.5f));

        cam.image_width = image_width;
        cam.image_height = image_height;
        cam.samples_per_pixel = samples;
        cam.max_depth = max_depth;

        cam.center = look_from;
        cam.pixel00_loc = pixel00_loc;
        cam.pixel_delta_u = pixel_delta_u;
        cam.pixel_delta_v = pixel_delta_v;

        cam.defocus_disk_u = mul3(u, defocus_radius);
        cam.defocus_disk_v = mul3(v, defocus_radius);
        cam.defocus_angle = defocus_angle;

        return cam;
    }

    struct SceneData
    {
        GpuCamera camera;
        std::vector<GpuSphere> spheres;
        std::vector<GpuMaterial> materials;
    };

    SceneData setup_simple_scene(int samples)
    {
        SceneData s{};
        s.materials.reserve(5);

        int ground = static_cast<int>(s.materials.size());
        //s.materials.push_back(make_metal(make_vec3(0.5f, 0.5f, 0.5f), 0.1f));

        int center = static_cast<int>(s.materials.size());
        s.materials.push_back(make_lambertian(make_vec3(0.1f, 0.2f, 0.5f)));

        int left = static_cast<int>(s.materials.size());
        //s.materials.push_back(make_dielectric(1.5f));

        int bubble = static_cast<int>(s.materials.size());
        //s.materials.push_back(make_dielectric(1.0f / 1.5f));

        int right = static_cast<int>(s.materials.size());
        //s.materials.push_back(make_metal(make_vec3(0.8f, 0.6f, 0.2f), 1.0f));

        add_sphere(s.spheres, make_vec3(0.0f, -100.5f, -1.0f), 100.0f, ground);
        add_sphere(s.spheres, make_vec3(0.0f, 0.0f, -1.2f), 0.5f, center);
        add_sphere(s.spheres, make_vec3(-1.0f, 0.0f, -1.0f), 0.5f, left);
        add_sphere(s.spheres, make_vec3(-1.0f, 0.0f, -1.0f), 0.4f, bubble);
        add_sphere(s.spheres, make_vec3(1.0f, 0.0f, -1.0f), 0.5f, right);

        int image_width = 400;
        float aspect_ratio = 16.0f / 9.0f;
        int max_depth = 50;
        float vfov = 40.0f;
        float3 look_from = make_vec3(0.0f, 0.0f, 1.0f);
        float3 look_at = make_vec3(0.0f, 0.0f, -1.0f);
        float3 vup = make_vec3(0.0f, 1.0f, 0.0f);
        float defocus_angle = 10.0f;
        float focus_dist = len3(sub3(look_from, look_at));

        s.camera = build_camera(
            image_width,
            aspect_ratio,
            samples,
            max_depth,
            vfov,
            look_from,
            look_at,
            vup,
            defocus_angle,
            focus_dist);

        return s;
    }

    SceneData setup_cover_scene(int samples)
    {
        SceneData s{};
        std::mt19937 rng(1337u);

        int ground = static_cast<int>(s.materials.size());
        s.materials.push_back(make_lambertian(make_vec3(0.5f, 0.5f, 0.5f)));
        add_sphere(s.spheres, make_vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground);

        for (int a = -11; a < 11; ++a)
        {
            for (int b = -11; b < 11; ++b)
            {
                float choose_mat = randf(rng);
                float3 center = make_vec3(static_cast<float>(a) + 0.9f * randf(rng), 0.2f, static_cast<float>(b) + 0.9f * randf(rng));

                float3 avoid = make_vec3(4.0f, 0.2f, 0.0f);
                if (len3(sub3(center, avoid)) <= 0.9f)
                {
                    continue;
                }

                if (choose_mat < 0.8f)
                {
                    int mat = static_cast<int>(s.materials.size());
                    s.materials.push_back(make_lambertian(mul3(random_color(rng), random_color(rng))));
                    add_sphere(s.spheres, center, 0.2f, mat);
                }
                else if (choose_mat < 0.95f)
                {
                    int mat = static_cast<int>(s.materials.size());
                    s.materials.push_back(make_metal(random_color(rng, 0.5f, 1.0f), randf(rng, 0.0f, 0.5f)));
                    add_sphere(s.spheres, center, 0.2f, mat);
                }
                else
                {
                    int mat = static_cast<int>(s.materials.size());
                    s.materials.push_back(make_dielectric(1.5f));
                    add_sphere(s.spheres, center, 0.2f, mat);
                }
            }
        }

        int m1 = static_cast<int>(s.materials.size());
        s.materials.push_back(make_dielectric(1.5f));
        add_sphere(s.spheres, make_vec3(0.0f, 1.0f, 0.0f), 1.0f, m1);

        int m2 = static_cast<int>(s.materials.size());
        s.materials.push_back(make_lambertian(make_vec3(0.4f, 0.2f, 0.1f)));
        add_sphere(s.spheres, make_vec3(-4.0f, 1.0f, 0.0f), 1.0f, m2);

        int m3 = static_cast<int>(s.materials.size());
        s.materials.push_back(make_metal(make_vec3(0.7f, 0.6f, 0.5f), 0.0f));
        add_sphere(s.spheres, make_vec3(4.0f, 1.0f, 0.0f), 1.0f, m3);

        s.camera = build_camera(
            1200,
            16.0f / 9.0f,
            samples,
            2,
            20.0f,
            make_vec3(13.0f, 2.0f, 3.0f),
            make_vec3(0.0f, 0.0f, 0.0f),
            make_vec3(0.0f, 1.0f, 0.0f),
            0.6f,
            10.0f);

        return s;
    }

} // namespace

int main(int argc, char **argv)
{
    std::string scene_name = "cover";
    int samples = 10;

    if (argc >= 2)
    {
        scene_name = argv[1];
    }
    if (argc >= 3)
    {
        try
        {
            samples = std::stoi(argv[2]);
            if (samples < 1)
            {
                std::cerr << "samples must be >= 1\n";
                return 1;
            }
        }
        catch (const std::exception &)
        {
            std::cerr << "Invalid samples value: " << argv[2] << " (must be integer)\n";
            return 1;
        }
    }

    SceneData scene;
    if (scene_name == "cover")
    {
        scene = setup_cover_scene(samples);
    }
    else if (scene_name == "simple")
    {
        scene = setup_simple_scene(samples);
    }
    else
    {
        std::cerr << "Invalid scene: " << scene_name << " (use cover|simple)\n";
        return 1;
    }

    double cuda_seconds = 0.0;
    std::string cuda_error;
    std::string output = get_output_filename(scene_name);

    if (!render_cuda_scene(scene.camera, scene.spheres, scene.materials, output, cuda_seconds, cuda_error))
    {
        std::cerr << "CUDA render failed: " << cuda_error << "\n";
        return 1;
    }

    std::cout << "CUDA render time: " << cuda_seconds << " s\n";
    return 0;
}
