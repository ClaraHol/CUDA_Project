
#include "rt_weekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"


hittable_list make_simple_world(){
     // World
    hittable_list world;

    auto material_ground = make_shared<metal>(color(0.5, 0.5, 0.5), 0.1);
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = make_shared<dielectric>(1.5);
    auto material_bubble = make_shared<dielectric>(1.00/1.50);
    auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100, material_ground));
    world.add(make_shared<sphere>(point3(0, 0, -1.2), 0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.4, material_bubble));
    world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    world = hittable_list(make_shared<bvh_node>(world));
    return world;
}

hittable_list make_complex_world(){
    // Coverpage from book
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_float();
            point3 center(a + 0.9*random_float(), 0.2, b + 0.9*random_float());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.7) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.85) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_float(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.98) {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                     // glass
                    sphere_material = make_shared<diffuse_light>(color(4.0, 4.0, 4.0));
                    world.add(make_shared<sphere>(center + point3(0, 3, 0), 0.2, sphere_material));

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

    world = hittable_list(make_shared<bvh_node>(world));
    return world;
}

int main() {
    

    camera cam;

    camera cam;
    int samples_per_pixels[9] = {100, 150, 200, 250, 300, 350, 400, 450, 500};
    for (int j=0; j<9; j++){

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1200;
    cam.samples_per_pixel = samples_per_pixels[j];
    cam.max_depth         = 10;

    cam.vfov     = 20;
    cam.look_from = point3(13,2,3);
    cam.look_at   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    hittable_list world;
    world = make_complex_world();


    auto t = omp_get_wtime();
    cam.render(world);
    t = omp_get_wtime() - t;

    std::clog << "\rSequential time: " << t << "\n";


    int num_threads[7] = {2, 4, 8, 16, 32, 64, 72}; 

    for ( int i=0; i<7; i++){
        omp_set_num_threads(num_threads[i]);
        t = omp_get_wtime();
        cam.render_parallel(world);
        t = omp_get_wtime() - t;
        std::clog << "\rParallel(" << omp_get_max_threads() << ")" << " time:" << t << "\n";

    }
    }
}
