
#include "rt_weekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)

void write_png(color* framebuffer, int num_pixels, int samples_per_pixel, int width, int height, std::string output_name){
    std::vector<uint8_t>pixels(num_pixels * 3);
    float normalizer = 1.0f/samples_per_pixel;
    for (int i = 0; i < num_pixels; i++){
        color c = write_color(framebuffer[i]*normalizer);
        pixels[i*3 + 0] = c[0];
        pixels[i*3 + 1] = c[1];
        pixels[i*3 + 2] = c[2];

    }
    stbi_write_png(output_name.c_str(), width, height, 3, pixels.data(), width*3);
}


__global__ void create_very_simple_world(hittable** d_list, hittable** d_world){
    /* 
        Create a simple scene with 1 sphere to illustrate the different materials
    */

    // Define world on host using one thread only
    if (threadIdx.x == 0 && blockIdx.x == 0){


        // Add spheres to world (sphere(center, radius, material))
        d_list[0] = new sphere(point3(0, -100.5, -1), 100, new lambertian(color(0.5, 0.5, 0.5)));
        d_list[1] = new sphere(point3(1.0, 0.5, -1.0), 1.0, new diffuse_light(color(8.0, 1.0, 0.0)));   
        d_list[2] = new sphere(point3(1.0, 0.5, 1.5), 1.0, new diffuse_light(color(0.0, 2.0, 4.0)));                  
      


        // Create world
        *d_world = new hittable_list(d_list, 5);
    }
};

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
                        vec3 albedo = vec3::lcg_random_float(state) * vec3::lcg_random_float(state);
                        d_list[i++] = new sphere(center, 0.2, new lambertian(albedo));

                        
                    } else if (choose_mat < 0.87){
                        // Metal
                        vec3 albedo = vec3::lcg_random_float(0.5, 1, state);
                        float fuzz = lcg_random(0, 0.5, state);
                        d_list[i++] = new sphere(center, 0.2, new metal(albedo, fuzz));
                
                        
                    } else if (choose_mat < 0.98) {
                        // Glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    } else {
                        d_list[i++] = new sphere(center + point3(0, 3, 0), 0.2f, new diffuse_light(color(4.0, 4.0, 4.0)));
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
    //d_list[i++] = new sphere(point3(0, 3, 0), 1.0, new diffuse_light(color(1.0, 1.0, 1.0)));
        
    // Create world
    *d_world = new hittable_list(d_list, i); 

    }
};


__global__ void render(
                    bvh_array* bvh_flat, hittable** d_list, color* framebuffer, int width, int height,
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
        pixel_color += cam -> ray_color(r,  d_list, bvh_flat, state); 
    }

    framebuffer[idx] = pixel_color * (1.0/float(samples_per_pixel));
}

__host__ void write_png(color* framebuffer, int num_pixels, int image_width, int image_height, std::string output_name){
    std::vector<uint8_t>pixels(num_pixels * 3);
    for (int i=0; i < num_pixels; i++){
        color c = write_color(framebuffer[i]);
        pixels[i*3 + 0] = c[0];
        pixels[i*3 + 1] = c[1];
        pixels[i*3 + 2] = c[2];

    }
    stbi_write_png(output_name.c_str(), image_width, image_height, 3, pixels.data(), image_width*3);
}

// DEBUGGING FUNCTION
__global__ void check_objects(hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("num_objects = %d\n", num_objects);
        for (int i = 0; i < num_objects; i++) {
            printf("objects[%d] = %p\n", i, objects[i]);
            if (objects[i] != nullptr) {
                aabb box = objects[i]->bounding_box();
                printf("  bbox x=(%f, %f)\n", box.x.min, box.x.max);
            }
        }
    }
}


int main(){

    // Set memory limits
    //cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);        // 64KB stack per thread
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512 * 1024 * 1024); // 512MB heap


    // World
    float aspect_ratio = 16.0f/9.0f;
    int num_objects = 5;
    int image_width = 1200;
    int image_height = int(float(image_width)/aspect_ratio);
    int samples_per_pixel = 100;
    int max_depth = 10;

    image_height = (image_height < 1) ? 1 : image_height;           // Ensure that image height is at least 1.

    int num_pixels = image_width * image_height;  


    dim3 dimBlock(16, 16);
    dim3 dimGrid((image_width + dimBlock.x -1)/dimBlock.x, (image_height + dimBlock.y -1)/dimBlock.y);

    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, num_pixels * sizeof(curandState)));
    unsigned long seed = 9999999;
    
    

    // Init random seeds for each thread
    init_rand<<<dimGrid, dimBlock>>>(d_states, image_width, image_height, seed);
    cudaDeviceSynchronize();
    

    // Make World
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
    

    create_simple_world<<<1, 1>>>(d_list, d_world);
    CUDA_CHECK(cudaDeviceSynchronize());

        
    // Make BVH
    bvh_array* d_bvh;
    int* d_bvh_size;

    CUDA_CHECK(cudaMalloc(&d_bvh, 4*num_objects*sizeof(bvh_array)));
    cudaMalloc(&d_bvh_size, sizeof(int));

    build_bvh<<<1,1>>>(d_list, num_objects, d_bvh, d_bvh_size);
    cudaDeviceSynchronize();

    // Check that BVH build was succesfull
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "build_bvh failed: " << cudaGetErrorString(err) << "\n";



    // Make camera
    camera cam;

    // Set viewport
    cam.aspect_ratio = aspect_ratio;
    cam.image_width = image_width;

   
    
    // Control camera position  (Camera for the complex scene)
    cam.vfov     = 20;
    cam.look_from = point3(13,2,3);
    cam.look_at   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.0;
    cam.focus_dist    = 10.0;

    camera* d_cam;

    int samples_per_pixels[7] = {100, 450, 550, 600, 650, 700, 750};
    int max_depths[11] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; 
    for (int j=0; j<1; j++){
        samples_per_pixel = samples_per_pixels[j];
        //max_depth = max_depths[j];
        cam.samples_per_pixel = samples_per_pixel;        // Sampling to make smoother edges
        cam.max_depth = max_depth;                               // Maximum recursion depth

        d_cam = cam.move_to_device();
        std::clog<< "Max depth: " << max_depth << "\n";
        //std::clog<< "Max depth: " << max_depth << "\n";


        int N = 10;

         cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float t = 0;

        
        cudaEventRecord(start, 0);
        for (int i = 0; i<N; i++){
            render<<<dimGrid, dimBlock>>>(d_bvh, d_list, d_framebuffer, image_width, image_height, samples_per_pixel, max_depth, d_cam, d_states);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        std::clog << "\r Average Cuda rendering time (" << N << " runs): " << t/float(N) << "\n";
    }

    cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    
    std::string output_name = "simple_scene.png";
    // Write to png
    write_png(h_framebuffer, num_pixels, image_width, image_height, output_name);


    cudaFree(d_bvh);
    cudaFree(d_framebuffer);
    cudaFree(d_cam);
    cudaFree(d_states);
    cudaFree(d_list); 
    cudaFree(d_world);
    cudaFreeHost(h_framebuffer);

    cudaDeviceReset();  // forces printf buffer flush
}
  
