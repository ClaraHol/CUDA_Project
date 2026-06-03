
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

    __host__ void write_png(color* framebuffer, int num_pixels, int width, int height, std::string output_name){
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
        d_list[0] = new sphere(point3(0.0f, -100.5f, -1.0f), 100.0f, new metal(color(0.5f, 0.5f, 0.5f), 0.1f));   // Ground ball
        d_list[1] = new sphere(point3(0.0f, 0.0f, -1.2f), 0.5f, new lambertian(color(0.1f, 0.2f, 0.5f)));      // Center ball
        d_list[2] = new sphere(point3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5f));                  // Hollow Glass ball 
        d_list[3] = new sphere(point3(-1.0f, 0.0f, -1.0f), 0.4f, new dielectric(1.00f/1.5f));             // Air inside Glass ball
        d_list[4] = new sphere(point3(1.0f, 0.0f, -1.0f), 0.5f, new metal(color(0.8f, 0.6f, 0.2f), 0.8f));  // Right metal ball (matte)

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
        d_list[0] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, new lambertian(color(0.5f, 0.5f, 0.5f)));

        // Make the random balls
        int i = 1;

        for (int a = -11; a < 11; a++){
            for (int b = -11; b < 11; b++){
                
                float choose_mat = lcg_random(state);
                point3 center(a + 0.9f * lcg_random(state), 0.2f, b + 0.9f *lcg_random(state));

                if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f){
                    
                    if (choose_mat < 0.7){
                        // Diffuse
                        vec3 albedo = vec3::lcg_random_float(state) * vec3::lcg_random_float(state);
                        d_list[i++] = new sphere(center, 0.2, new lambertian(albedo));

                        
                    } else if (choose_mat < 0.87){
                        // Metal
                        vec3 albedo = vec3::lcg_random_float(0.5f, 1.0f, state);
                        float fuzz = lcg_random(0.0f, 0.5f, state);
                        d_list[i++] = new sphere(center, 0.2f, new metal(albedo, fuzz));
                
                        
                    } else if (choose_mat < 0.98) {
                        // Glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    } else {
                        d_list[i++] = new sphere(center + point3(0, 3, 0), 0.2f, new diffuse_light(color(4.0f, 4.0f, 4.0f)));
                    }  
                }else{
                    b--;
                }
            }
        }

    // Make the big spheres
    d_list[i++] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    d_list[i++] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(color(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, new metal(color(0.7f, 0.6f, 0.5f), 0.0));
        
    // Create world
    *d_world = new hittable_list(d_list, i); 
    }
}


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
            float normalizer = 1.0f/float(samples_per_pixel);
            
            
            // Render
            color pixel_color(0.0f, 0.0f, 0.0f);
            for (int k = 0; k < samples_per_pixel; k++){
                ray r  = cam -> get_ray(i, j, state);
                pixel_color += cam -> ray_color(r, max_depth, world, state); 
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
    float aspect_ratio = 16.0/9.0;
    int num_objects = 488;
    int image_width = 1200;
    int image_height = int(float(image_width)/aspect_ratio);
    int samples_per_pixel = 100;
    int max_depth = 10;

    image_height = (image_height < 1) ? 1 : image_height;           // Ensure that image height is at least 1.

    int num_pixels = image_width * image_height;  

    int samples_per_pixels[7] = {100, 450, 550, 600, 650, 700, 750};
    int max_depths[11] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; 
    
    for (int j=0; j<1; j++){
        //max_depth = max_depths[j];
        samples_per_pixel = samples_per_pixels[j]; 
        dim3 dimBlock(16, 16);
        dim3 dimGrid((image_width + dimBlock.x -1)/dimBlock.x, (image_height + dimBlock.y -1)/dimBlock.y);

        curandState* d_states;
        CUDA_CHECK(cudaMalloc(&d_states, num_pixels * sizeof(curandState)));
        unsigned long seed = 123456789;
        
        // Make sure that cuda can allocate enough memory
        cudaDeviceSetLimit(cudaLimitStackSize, 32768);  // 32KB per thread
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * 1024 * 1024)); // 512 MB


        init_rand<<<dimGrid, dimBlock>>>(d_states, image_width, image_height, seed);
        cudaDeviceSynchronize();
        
        hittable** d_list;
        hittable** d_world;
        color*    d_framebuffer;
        color*    h_framebuffer;

        CUDA_CHECK(cudaMalloc(&d_list, num_objects * sizeof(hittable*)));
        CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable*)));
        CUDA_CHECK(cudaMalloc(&d_framebuffer, num_pixels * sizeof(color)));
        CUDA_CHECK(cudaMallocHost(&h_framebuffer, num_pixels * sizeof(color)));

        // Create world
        create_complex_world<<<1, 1>>>(d_list, d_world);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        camera cam;

        // Set viewport
        cam.aspect_ratio = aspect_ratio;
        cam.image_width = image_width;
        cam.samples_per_pixel = samples_per_pixel;        // Sampling to make smoother edges
        cam.max_depth = max_depth;                               // Maximum recursion depth
        
        // Camera for the complex scene
        cam.vfov     = 20;
        cam.look_from = point3(13,2,3);
        cam.look_at   = point3(0,0,0);
        cam.vup      = vec3(0,1,0);

        cam.defocus_angle = 0.0;
        cam.focus_dist    = 10.0;

        camera* d_cam = cam.move_to_device();

        auto t = omp_get_wtime();
        int N = 10;

        for (int i = 0; i<N; i++){
            //std::clog<< "Iteration: " << i+1 << "\n";
            render<<<dimGrid, dimBlock>>>(d_world, d_framebuffer, image_width, image_height, samples_per_pixel, max_depth, d_cam, d_states);
            cudaDeviceSynchronize();
        }
        t = omp_get_wtime() - t;
        std::clog << "\r Average Cuda rendering time (" << N << " runs): " << t/float(N) << "\n";

        cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);

        std::string output_name = "complex_image.png";
        // Write to png
        write_png(h_framebuffer, num_pixels, image_width, image_height, output_name);

        
    
        cudaFree(d_list);
        cudaFree(d_world);
        cudaFree(d_framebuffer);
        cudaFree(d_cam);
        cudaFree(d_states);
        cudaFreeHost(h_framebuffer);

        cudaDeviceReset();  // forces printf buffer flush
    }
}
  
