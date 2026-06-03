
#include "rt_weekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"             // To write png image

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)

__host__ void write_png(color* framebuffer, int num_pixels, int samples_per_pixel, int width, int height, std::string output_name){
    std::vector<uint8_t>pixels(num_pixels * 3);
    for (int i = 0; i < num_pixels; i++){
        color c = write_color(framebuffer[i]);
        pixels[i*3 + 0] = c[0];
        pixels[i*3 + 1] = c[1];
        pixels[i*3 + 2] = c[2];

    }
    stbi_write_png(output_name.c_str(), width, height, 3, pixels.data(), width*3);
}


__global__ void create_simple_world(hittable** d_list, hittable** d_world){
    /* 
        Create a simple scene with 4 balls.
    */

    // Define world on host using one thread only
    if (threadIdx.x == 0 && blockIdx.x == 0){


        // Add spheres to world (sphere(center, radius, material))
        d_list[0] = new sphere(point3(0, -100.5, -1), 100, new metal(color(0.5, 0.5, 0.5), 0.1));   // Ground ball
        d_list[1] = new sphere(point3(0, 0, -1.2), 0.5, new lambertian(color(0.1, 0.2, 0.5)));      // Center ball
        d_list[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, new dielectric(1.5));                  // Hollow Glass ball 
        d_list[3] = new sphere(point3(-1.0, 0.0, -1.0), 0.4, new dielectric(1.00/1.5));             // Air inside Glass ball
        d_list[4] = new sphere(point3(1.0, 0.0, -1.0), 0.5, new metal(color(0.8, 0.6, 0.2), 0.8));  // Right metal ball (matte)

        // Create world
        *d_world = new hittable_list(d_list, 5);
    }
};


__global__ void create_complex_world(hittable** d_list, hittable** d_world){
    /*
        Create the complex scene with 3 big balls and lots of smaller balls.
    */
   if (threadIdx.x == 0 && blockIdx.x == 0){
        uint32_t  state = 1234567;  // seed
        

        // Make the ground
        d_list[0] = new sphere(point3(0, -1000, 0), 1000, new lambertian(color(0.5, 0.5, 0.5)));

        // Make the random balls
        int i = 1;
        for (int a = -11; a < 11; a++){
            for (int b = -11; b < 11; b++){
                
                float choose_mat = lcg_random(state);
                point3 center(a + 0.9 * lcg_random(state), 0.2, b + 0.9 *lcg_random(state));

                if ((center - point3(4, 0.2, 0)).length() > 0.9){
                    
                    if (choose_mat < 0.7){
                        // Diffuse
                        vec3 albedo = vec3::lcg_random_double(state) * vec3::lcg_random_double(state);
                        d_list[i++] = new sphere(center, 0.2, new lambertian(albedo));

                        
                    } else if (choose_mat < 0.85){
                        // Metal
                        vec3 albedo = vec3::lcg_random_double(0.5, 1, state);
                        float fuzz = lcg_random(0, 0.5, state);
                        d_list[i++] = new sphere(center, 0.2, new metal(albedo, fuzz));
                
                        
                   } else if (choose_mat < 0.98) {
                        // Glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    } else {
                        d_list[i++] = new sphere(center + point3(0, 3, 0), 0.2, new diffuse_light(color(4.0, 4.0, 4.0)));
                    }  
                }else{
                    b--;
                }
            }
        }

    // Make the big spheres
    d_list[i++] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
    d_list[i++] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
    d_list[i++] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));
        
    // Create world
    *d_world = new hittable_list(d_list, i); 
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
            double normalizer = 1.0/double(samples_per_pixel);
            
            
            // Render
            color pixel_color(0.0, 0.0, 0.0);
            for (int k = 0; k < samples_per_pixel; k++){
                ray r  = cam -> get_ray(i, j, state);
                pixel_color += cam -> ray_color_iterative(r, world, state); 
            }

            framebuffer[idx] = pixel_color * normalizer;
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
    int num_objects = 488;
    int image_width = 1200;
    int image_height = int(double(image_width)/aspect_ratio);
    int samples_per_pixel = 500;
    int max_depth = 10;

    image_height = (image_height < 1) ? 1 : image_height;           // Ensure that image height is at least 1.

    int num_pixels = image_width * image_height;  
    int samples_per_pixels[10] = {1, 50, 100, 150, 200, 250, 300, 350, 400, 500};
    int max_depths[11] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; 
    
    for (int j=0; j<11; j++){
        max_depth = max_depths[j];

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

        create_complex_world<<<1, 1>>>(d_list, d_world);
        CUDA_CHECK(cudaDeviceSynchronize());
    

        camera cam;

        // Set viewport
        cam.aspect_ratio = aspect_ratio;
        cam.image_width = image_width;
        cam.samples_per_pixel = samples_per_pixel;        // Sampling to make smoother edges
        cam.max_depth = max_depth;                               // Maximum recursion depth
        
        // Control camera position
        cam.vfov     = 20;
        cam.look_from = point3(13,2,3);
        cam.look_at   = point3(0,0,0);
        cam.vup      = vec3(0,1,0);

        cam.defocus_angle = 0.0;
        cam.focus_dist    = 10.0;

        camera* d_cam = cam.move_to_device();

        auto t = omp_get_wtime();
        int N = 10;

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float t = 0;

        
        cudaEventRecord(start, 0);
        for (int i = 0; i<N; i++){
            //std::clog<< "Iteration: " << i+1 << "\n";
            render<<<dimGrid, dimBlock>>>(d_world, d_framebuffer, image_width, image_height, samples_per_pixel, max_depth, d_cam, d_states);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        std::clog << "\r Average Cuda rendering time (" << N << " runs): " << t/float(N) << "\n";


        cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);

        std::clog<< "Writting image to png... \n";
        std::string output_name = "complex_image.png";
        // Write to png
        write_png(h_framebuffer, num_pixels, samples_per_pixel, image_width, image_height, output_name);
        std::clog<< "Done! \n";

        
        // Free allocated arrays
        cudaFree(d_list);
        cudaFree(d_world);
        cudaFree(d_framebuffer);
        cudaFree(d_cam);
        cudaFree(d_states);
        cudaFreeHost(h_framebuffer);

        cudaDeviceReset();  // forces printf buffer flush
    }
}
  