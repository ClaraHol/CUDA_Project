
#include "rt_weekend.h"

#include "bvh.h"
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
__global__ void initRandKernel(curandState* states, int width, int height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= width || j >= height) return;

    int idx = i + j * width;

    // Each thread/ray gets a unique sequence to avoid correlation
    curand_init(seed, idx, 0, &states[idx]);
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

                if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9){
                    
                    if (choose_mat < 0.8f){
                        // Diffuse
                        vec3 albedo = vec3::lcg_random_float(state) * vec3::lcg_random_float(state);
                        d_list[i++] = new sphere(center, 0.2, new lambertian(albedo));

                        
                    } else if (choose_mat < 0.95f){
                        // Metal
                        vec3 albedo = vec3::lcg_random_float(0.5f, 1.0f, state);
                        float fuzz = lcg_random(0.0f, 0.5f, state);
                        d_list[i++] = new sphere(center, 0.2f, new metal(albedo, fuzz));
                
                        
                    } else {
                        // Glass
                        d_list[i++] = new sphere(center, 0.2f, new dielectric(1.5f));
                    }  
                }else{
                    b--;
                }
            }
        }

    // Make the big spheres
    d_list[i++] = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, new dielectric(1.5f));
    d_list[i++] = new sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, new lambertian(color(0.4f, 0.2f, 0.1f)));
    d_list[i++] = new sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, new metal(color(0.7f, 0.6f, 0.5f), 0.0f));
        
    // Create world
    *d_world = new hittable_list(d_list, i); 
    }
};

// Old rendering kernel
__global__  void render(
    bvh_array* bvh_flat, hittable** d_list,  uchar4* pbo, int width, int height, int samples_per_pixel, int max_depth, int frame, camera* cam, curandState* states) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int idx = i + (height - j - 1) * width; // Flip image to match with GLEWs standard.

    curandState* state = &states[idx];

    color pixel_color(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < samples_per_pixel; k++){
        ray r = cam->get_ray(i, j, state);
        pixel_color += cam->ray_color(r, max_depth, d_list, bvh_flat, state);
    }
    pixel_color *= 1.0f / float(samples_per_pixel);

    // Gamma correct and clamp inline — no function call needed
    auto r = __saturatef(sqrtf(pixel_color.x()));  // __saturatef clamps to [0,1]
    auto g = __saturatef(sqrtf(pixel_color.y()));
    auto b = __saturatef(sqrtf(pixel_color.z()));

    pbo[idx] = make_uchar4(
        (unsigned char)(255.99f * r),
        (unsigned char)(255.99f * g),
        (unsigned char)(255.99f * b),
        255
    );
}

// New rendering kernels split into two to make rendering iteratively better when camera is still

// Kernel 1 — add one sample to accumulation buffer
__global__ void renderAccum(bvh_array* bvh, hittable** d_list,
                             float3* accum, int width, int height,
                             int max_depth, int samples_per_pixel, camera* cam, curandState* states)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int idx = i + (height - 1 - j) * width;

    color c(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < samples_per_pixel; k++){
            ray r = cam->get_ray(i, j, &states[idx]);
            c += cam->ray_color(r, max_depth, d_list, bvh,  &states[idx]);
        }
        c *= 1.0f / float(samples_per_pixel);

    accum[idx].x += c.x();
    accum[idx].y += c.y();
    accum[idx].z += c.z();
}

// Kernel 2 — resolve accumulation buffer to uchar4 PBO
__global__ void resolve(float3* accum, uchar4* pbo,
                        int width, int height, int frame)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int idx = i + (height - 1 - j) * width;

    float scale = 1.0f / float(frame);
    float r = __saturatef(sqrtf(accum[idx].x * scale));
    float g = __saturatef(sqrtf(accum[idx].y * scale));
    float b = __saturatef(sqrtf(accum[idx].z * scale));

    pbo[idx] = make_uchar4(
        (unsigned char)(255.99f * r),
        (unsigned char)(255.99f * g),
        (unsigned char)(255.99f * b),
        255
    );
}
void initRand(curandState* states, int width, int height, unsigned long seed){
    dim3 dimBlock(16, 8);
    dim3 dimGrid((width + dimBlock.x -1)/dimBlock.x, (height + dimBlock.y -1)/dimBlock.y);
    initRandKernel<<<dimGrid, dimBlock>>>(states, width, height, seed);
    cudaDeviceSynchronize();
}

void initWorld(bvh_array* d_bvh, hittable** d_list, hittable** d_world, int* d_bvh_size, int num_objects, int width, int height){
    // Make the world and BVH  
    create_complex_world<<<1, 1>>>(d_list, d_world);
    cudaDeviceSynchronize();

    build_bvh<<<1,1>>>(d_list, num_objects, d_bvh, d_bvh_size);
    cudaDeviceSynchronize();
}

void initCamera(camera& h_cam, camera*& d_cam, int image_height, int image_width, int samples_pp, int max_depth, float vfov, point3 look_from, point3 look_at, vec3 vup, float defocus_angle, float focus_dist){

    // Set viewport
    h_cam.image_width = image_width;
    h_cam.image_height = image_height;
    h_cam.samples_per_pixel = samples_pp;                      // Sampling to make smoother edges
    h_cam.max_depth = max_depth;                               // Maximum recursion depth
    
    // Control camera position  (Camera for the complex scene)
    h_cam.vfov      = vfov;
    h_cam.look_from = look_from;
    h_cam.look_at   = look_at;
    h_cam.vup       = vup;

    h_cam.defocus_angle = defocus_angle;
    h_cam.focus_dist    = focus_dist;

    h_cam.move_to_device(d_cam);
}

void launchRayTracer(bvh_array* bvh_flat, hittable** d_list, uchar4* pbo, int width, int height,
                    int samples_per_pixel, int max_depth, int frame, camera* cam, curandState* states, 
                    cudaStream_t stream){
    
    dim3 dimBlock(16, 8);
    dim3 dimGrid((width + dimBlock.x -1)/dimBlock.x, (height + dimBlock.y -1)/dimBlock.y);
    
    render<<<dimGrid, dimBlock, 0, stream>>>(bvh_flat, d_list, pbo, width, height, samples_per_pixel, max_depth, frame, cam, states);
    cudaDeviceSynchronize();
} 

void launchAccumRayTracer(bvh_array* bvh_flat, hittable** d_list, float3* accum, uchar4* pbo, int width, int height,
                    int max_depth, int samples_per_pixel, int frame, camera* cam, curandState* states, cudaStream_t stream){
    dim3 dimBlock(16, 8);
    dim3 dimGrid((width + dimBlock.x -1)/dimBlock.x, (height + dimBlock.y -1)/dimBlock.y);
    
    renderAccum<<<dimGrid, dimBlock, 0, stream>>>(bvh_flat, d_list, accum, width, height, max_depth, samples_per_pixel,  cam, states);

    resolve<<<dimGrid, dimBlock, 0, stream>>>(accum, pbo, width, height, frame);
    cudaDeviceSynchronize();
}
