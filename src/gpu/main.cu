
#include "rt_weekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)


__global__ void create_simple_world(hittable** d_list, hittable** d_world, material** d_material){
    /* 
        Create the Scene. Note that this has to be changed to change the scene
    */

    // Define world on host using one thread only
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // Define materials
        d_material[0] = new metal(color(0.5, 0.5, 0.5), 0.1);    // Ground ball
        d_material[1] = new lambertian(color(0.1, 0.2, 0.5));    // Center ball
        d_material[2] = new dielectric(1.5);                     // Hollow Glass ball 
        d_material[3] = new dielectric(1.00/1.5);                // Air inside Glass ball
        d_material[4] = new metal(color(0.8, 0.6, 0.2), 0.8);    // Right metal ball (matte)


        // Add spheres to world (sphere(center, radius, material))
        d_list[0] = new sphere(point3(0, -100.5, -1), 100, d_material[0]);
        d_list[1] = new sphere(point3(0, 0, -1.2), 0.5, d_material[1]);
        d_list[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, d_material[2]);
        d_list[3] = new sphere(point3(-1.0, 0.0, -1.0), 0.4, d_material[3]);
        d_list[4] = new sphere(point3(1.0, 0.0, -1.0), 0.5, d_material[4]);

        // Create world
        *d_world = new hittable_list(d_list, 5);
    }
};

__global__ void free_world(hittable** d_list, hittable** d_world, material** d_material, int num_objects){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        for (int i = 0; i < num_objects; i++){
            delete d_list[i];
            delete d_material[i];
        }
        
        delete *d_world;
    }
};

 __global__ void render(
                    hittable**  world, color* framebuffer, int width, int height,
                    int samples_per_pixel, int max_depth, camera* cam, curandState* states
                    ){
            
            /* Render the image on the device */

            // Compute thread index
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            int j = threadIdx.y + blockIdx.y * blockDim.y;

            if (i >= width || j >= height) return;
            int idx = i + j * width;
            curandState* state = &states[idx];
            
            
            // Render
            color pixel_color(0, 0, 0);
            for (int k = 0; k < samples_per_pixel; k++){
                ray r  = cam -> get_ray(i, j, state);
                pixel_color += cam -> ray_color(r, max_depth, world, state); 
            }

            framebuffer[idx] = pixel_color * (1.0/double(samples_per_pixel));
        }
__global__ void debug_world(hittable** world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Try hitting with a ray pointing straight at the center sphere
        ray r(point3(0, 0, 3), vec3(0, 0, -1));
        hit_record rec;
        bool hit = (*world)->hit(r, interval(0.001, 1000), rec);
        printf("World hit test: %s\n", hit ? "HIT" : "MISS");
        printf("World ptr: %p\n", (void*)*world);
    }
}    

int main(){

    // World
    double aspect_ratio = 16.0/9.0;
    int num_objects = 5;
    int image_width = 400;
    int image_height = int(double(image_width)/aspect_ratio);
    int samples_per_pixel = 200;
    int max_depth = 50;

    image_height = (image_height < 1) ? 1 : image_height;           // Ensure that image height is at least 1.

    int num_pixels = image_width * image_height;  


    dim3 dimBlock(16, 16);
    dim3 dimGrid((image_width + dimBlock.x -1)/dimBlock.x, (image_height + dimBlock.y -1)/dimBlock.y);

    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, num_pixels * sizeof(curandState)));
    unsigned long seed = 123456789;
    
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);  // 32KB per thread
    init_rand<<<dimGrid, dimBlock>>>(d_states, image_width, image_height, seed);
    cudaDeviceSynchronize();
    
    hittable** d_list;
    hittable** d_world;
    material** d_material;
    color*    d_framebuffer;
    color*    h_framebuffer;

    CUDA_CHECK(cudaMalloc(&d_material, num_objects * sizeof(material*)));
    CUDA_CHECK(cudaMalloc(&d_list, num_objects * sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_framebuffer, num_pixels * sizeof(color)));
    CUDA_CHECK(cudaMallocHost(&h_framebuffer, num_pixels * sizeof(color)));

    create_simple_world<<<1, 1>>>(d_list, d_world, d_material);
    CUDA_CHECK(cudaDeviceSynchronize());
   

    camera cam;

    // Set viewport
    cam.aspect_ratio = aspect_ratio;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;        // Sampling to make smoother edges
    cam.max_depth = max_depth;                               // Maximum recursion depth
    
    // Control camera position
    cam.vfov = 40;                      // Distance from objects
    cam.look_from = point3(-2, 2, 1);    // Camera position
    cam.look_at = point3(0,0,-1);       // Point the camera looks at
    cam.vup = point3(0,1,0);            // Camera rotation

    cam.defocus_angle = 20.0;             //Defocusing
    cam.focus_dist = 3.6;

    camera* d_cam = cam.move_to_device();

    auto t = omp_get_wtime();
    render<<<dimGrid, dimBlock>>>(d_world, d_framebuffer, image_width, image_height, samples_per_pixel, max_depth, d_cam, d_states);
    cudaDeviceSynchronize();
    t = omp_get_wtime() - t;

    cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);

    // Write to std::cout to render image in cpu
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n"; // Image dimensions
    for (int j = 0; j < image_height; j++){
        for (int i = 0; i < image_width; i++){
            write_color(std::cout,  h_framebuffer[i + j*image_width]);
        }
    }

    std::clog << "\r Cuda rendering time: " << t << "\n";

    
    // Destroy the world
    free_world<<<1, 1>>>(d_list, d_world, d_material, num_objects);
    cudaDeviceSynchronize();

    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(d_material);
    cudaFree(d_framebuffer);
    cudaFree(d_cam);
    cudaFree(d_states);
    cudaFreeHost(h_framebuffer);

    cudaDeviceReset();  // forces printf buffer flush
}
  
/*
int main() {
    // Coverpage from book
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
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

    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1200;
    cam.samples_per_pixel = 500;
    cam.max_depth         = 50;

    cam.vfov     = 20;
    cam.look_from = point3(13,2,3);
    cam.look_at   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    auto t = omp_get_wtime();
    //cam.render(world);
    t = omp_get_wtime() - t;

    std::clog << "\rSequential time: " << t << "\n";

    cam.render_parallel(world);
}
*/
