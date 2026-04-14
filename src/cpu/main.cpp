
#include "rt_weekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include <string>
#include <stdexcept>

using namespace std;

#ifdef USE_CUDA
#include "../cuda/cuda_renderer.h"
#endif

struct Scene
{
    hittable_list world;
    camera cam;
};

// Generate output filename based on scene and mode
string get_output_filename(const string &scene_name, const string &mode)
{
    return "images/" + scene_name + "_" + mode + ".ppm";
};

// Coverpage from book with many spheres of different materials
Scene setup_coverpage_scene(int samples)
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8)
                {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95)
                {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 1200;
    cam.samples_per_pixel = samples;
    cam.max_depth = 50;

    cam.vfov = 20;
    cam.look_from = point3(13, 2, 3);
    cam.look_at = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    return {world, cam};
}

Scene setup_simple_scene(int samples)
{
    // World
    hittable_list world;

    auto material_ground = make_shared<metal>(color(0.5, 0.5, 0.5), 0.1);
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.5);
    auto material_bubble = make_shared<dielectric>(1.00 / 1.50);
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    // Simple scene with 3 spheres (one of each material type) and a ground plane.
    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100, material_ground));
    world.add(make_shared<sphere>(point3(0, 0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    camera cam;

    // Set viewport
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = samples; // Sampling to make smoother edges
    cam.max_depth = 50;              // Maximum recursion depth

    // Control camera position
    cam.vfov = 40;                   // Distance from objects
    cam.look_from = point3(0, 0, 1); // Camera position
    cam.look_at = point3(0, 0, -1);  // Point the camera looks at
    cam.vup = point3(0, 1, 0);       // Camera rotation

    cam.defocus_angle = 10; // Defocusing
    cam.focus_dist = 2.0;

    return {world, cam};
}

int main(int argc, char **argv)
{
    string mode;
    string scene_name;
    int samples = 10; // Default samples

    // Parse key/value args: scene <cover|simple> mode <cpu|omp|cuda|all> samples <integer>
    for (int idx = 1; idx < argc; idx += 2)
    {
        if (idx + 1 >= argc)
        {
            cerr << "Missing value for argument: " << argv[idx] << "\n";
            cerr << "Usage: ./raytrace scene <cover|simple> mode <cpu|omp|cuda|all> samples <integer>\n";
            return 1;
        }

        string key = argv[idx];
        string value = argv[idx + 1];

        if (key == "scene")
        {
            scene_name = value;
        }
        else if (key == "mode")
        {
            mode = value;
        }
        else if (key == "samples")
        {
            try
            {
                samples = stoi(value);
                if (samples < 1)
                {
                    cerr << "samples must be >= 1\n";
                    return 1;
                }
            }
            catch (const exception &e)
            {
                cerr << "Invalid samples value: " << value << " (must be integer)\n";
                return 1;
            }
        }
        else
        {
            cerr << "Unknown argument key: " << key << "\n";
            cerr << "Usage: ./raytrace scene <cover|simple> mode <cpu|omp|cuda|all> samples <integer>\n";
            return 1;
        }
    }

    // Defaults (if args are not passed)
    if (scene_name.empty())
    {
        scene_name = "cover";
    }
    if (mode.empty())
    {
        mode = "cuda";
    }

    Scene scene;

    // Scene selection
    if (scene_name == "cover")
    {
        scene = setup_coverpage_scene(samples);
    }
    else if (scene_name == "simple")
    {
        scene = setup_simple_scene(samples);
    }
    else
    {
        cerr << "Invalid scene: " << scene_name << " (use cover|simple)\n";
        return 1;
    }

    if (mode != "all" && mode != "cpu" && mode != "omp" && mode != "cuda")
    {
        cerr << "Invalid mode: " << mode << " (use cpu|omp|cuda|all)\n";
        return 1;
    }

    // Mode selection
    if (mode == "all" || mode == "cpu")
    {
        auto t = omp_get_wtime();
        scene.cam.render(scene.world, get_output_filename(scene_name, "cpu"));
        t = omp_get_wtime() - t;
        clog << "\rSequential time: " << t << "\n";
    }

    if (mode == "all" || mode == "omp")
    {
        scene.cam.render_parallel(scene.world, get_output_filename(scene_name, "omp"));
    }

    if (mode == "all" || mode == "cuda")
    {
#ifdef USE_CUDA
        double cuda_seconds = 0.0;
        string cuda_error;
        if (render_cuda_mvp(scene.cam, scene.world, get_output_filename(scene_name, "cuda"), cuda_seconds, cuda_error))
        {
            clog << "CUDA MVP time: " << cuda_seconds << "\n";
        }
        else
        {
            clog << "CUDA MVP failed: " << cuda_error << "\n";
            return 1;
        }
#else
        clog << "CUDA mode requested but binary was built without USE_CUDA.\n";
        return 1;
#endif
    }

    return 0;
}
