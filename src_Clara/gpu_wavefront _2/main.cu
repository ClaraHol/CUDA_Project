
#include "rt_weekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"             // To write png image

/*
Code Structure:
- Debugging functions
- Host functions
- World generation kernels
- Wavefront rendering kernels
- Kernel calling functions
- Main function 
*/


// ------------------------------------------------------------------------------------------------
// Debugging functions
// ------------------------------------------------------------------------------------------------

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)


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

// ------------------------------------------------------------------------------------------------
// Host functions
// ------------------------------------------------------------------------------------------------



__host__ void write_png(color* framebuffer, int num_pixels, int samples_per_pixel, int width, int height, std::string output_name){
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


// ------------------------------------------------------------------------------------------------
// Kernel Functions to generate the world
// ------------------------------------------------------------------------------------------------

__global__ void create_simple_world(hittable** d_list, hittable** d_world){
    /* 
        Create a simple scene with 4 balls.
    */

    // Define world on host using one thread only
    if (threadIdx.x == 0 && blockIdx.x == 0){


        // Add spheres to world (sphere(center, radius, material))
        d_list[0] = new sphere(point3(0, -100.5, -1), 100, new metal(0, color(0.5, 0.5, 0.5), 0.1));   // Ground ball
        d_list[1] = new sphere(point3(0, 0, -1.2), 0.5, new lambertian(1, color(0.1, 0.2, 0.5)));      // Center ball
        d_list[2] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, new dielectric(2, 1.5));                  // Hollow Glass ball 
        d_list[3] = new sphere(point3(-1.0, 0.0, -1.0), 0.4, new dielectric(3, 1.00/1.5));             // Air inside Glass ball
        d_list[4] = new sphere(point3(1.0, 0.0, -1.0), 0.5, new metal(4, color(0.8, 0.6, 0.2), 0.8));  // Right metal ball (matte)

        // Create world
        *d_world = new hittable_list(d_list, 5);
    }
};

__global__ void create_complex_world(material** d_material, hittable** d_list, hittable** d_world){
    /*
        Create the complex scene with 3 big balls and lots of smaller balls.
    */
   if (threadIdx.x == 0 && blockIdx.x == 0){
        uint32_t  state = 1234567;  // seed
        

        // Make the ground
        d_material[0] = new lambertian(0, color(0.5f, 0.5f, 0.5f));
        d_list[0] = new sphere(point3(0.0f, -1000.0f, 0.0f), 1000, d_material[0]);
        d_material[1] = new dielectric(1, 1.5f);

        // Make the random balls
        int i = 1;
        int j = 2;
        for (int a = -11; a < 11; a++){
            for (int b = -11; b < 11; b++){
                
                float choose_mat = lcg_random(state);
                point3 center(a + 0.9f * lcg_random(state), 0.2f, b + 0.9f *lcg_random(state));

                if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9){
                    
                    if (choose_mat < 0.7){
                        // Diffuse
                        vec3 albedo = vec3::lcg_random_float(state) * vec3::lcg_random_float(state);

                        d_material[j] = new lambertian(j, albedo);
                        d_list[i++] = new sphere(center, 0.2f, d_material[j]);
                        j++;

                        
                    } else if (choose_mat < 0.87){
                        // Metal
                        vec3 albedo = vec3::lcg_random_float(0.5f, 1, state);
                        float fuzz = lcg_random(0, 0.5f, state);

                        d_material[j] = new metal(j, albedo, fuzz);
                        d_list[i++] = new sphere(center, 0.2f, d_material[j]);
                        j++;
                
                        
                    } else if (choose_mat < 0.98f){
                        // Glass
                        d_list[i++] = new sphere(center, 0.2f, d_material[1]);

                    }  else {
                        d_material[j] = new diffuse_light(j, color(4.0f, 4.0f, 4.0f));  // bright white light
                        d_list[i++] = new sphere(center + point3(0, 3, 0), 0.2f, d_material[j]);
                        j++;
                    }
                }else{
                    b--;
                }
            }
        }
 
    // Make the big spheres
    d_list[i++] = new sphere(point3(0, 1, 0), 1.0, d_material[1]);

    d_material[j] = new lambertian(j, color(0.4, 0.2, 0.1)); 
    d_list[i++] = new sphere(point3(-4, 1, 0), 1.0, d_material[j]);
    j++;

    d_material[j] = new metal(j, color(0.7, 0.6, 0.5), 0.0);
    d_list[i++] = new sphere(point3(4, 1, 0), 1.0, d_material[j]);
        
    // Create world
    *d_world = new hittable_list(d_list, i); 
    }
};

// ------------------------------------------------------------------------------------------------
// Rendering Kernels in call order
// ------------------------------------------------------------------------------------------------

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

__global__ void init_rays(int samples_per_pixel, int width, int height, camera* cam, color* throughput, rayBuffer* rays, RngState* state){
    // Init all the rays
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    
    int idx = i + j * width;                                   //Image index

    for (int n = 0; n < samples_per_pixel; n++){
        int r_idx = n + idx * samples_per_pixel;               // Ray index                                        
        cam->make_ray(i, j, idx, r_idx, rays, &state[idx]);    // Make and store rays in ray buffer and ray index in hit buffer
        throughput[r_idx] = color(1.0f, 1.0f, 1.0f);
    }
    
}

__global__ void intersect(hitBuffer* hits, rayBuffer* rays, bvh_array* bvh, 
                           hittable** objects, int n_rays) 
    {
    // Should be ray indexes for contiguous access in hittable
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n_rays) return;
    if (!rays->alive[idx]) return;  // skip dead rays

    // Run the hit test (Handles all the writing to hits and rays in the sphere hit function)
    bvh[0].hit(bvh, objects, interval(0.001, infinity), hits, rays, idx);
}    


__global__ void classify(hitBuffer* hits, rayBuffer* rays,
                          MaterialQueue* queues, int n_rays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_rays) return;
    if (!rays->alive[idx]) return;

    if (!hits->hit_anything[idx]) {
        // Put in a "miss" queue or handle background here
        int pos = atomicAdd(&queues[MISS].count[0], 1);
        queues[MISS].ray_idx[pos] = idx;
        return;
    }

    MatType mat_type = (MatType)hits->mat_type[idx];
    if (mat_type < 0 || mat_type >= NUM_MAT_TYPES) return;  // guard against garbage
    int pos = atomicAdd(&queues[mat_type].count[0], 1);
    queues[mat_type].ray_idx[pos] = idx;
}

__global__ void shade_lambertian(MaterialQueue* queue, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, RngState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= queue[LAMBERTIAN].count[0]) return;

    int idx = queue[LAMBERTIAN].ray_idx[q_idx];
    lambertian* mat = (lambertian*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {
        rays->alive[idx] = 0;
    }
}


__global__ void shade_metal(MaterialQueue* queue, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, RngState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= queue[METAL].count[0]) return;

    int idx = queue[METAL].ray_idx[q_idx];
    metal* mat = (metal*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {
        rays->alive[idx] = 0;
    }

}

__global__ void shade_dielectric(MaterialQueue* queue, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, RngState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= queue[DIELECTRIC].count[0]) return;

    int idx = queue[DIELECTRIC].ray_idx[q_idx];
    dielectric* mat = (dielectric*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {
        rays->alive[idx] = 0;
    }
}

__global__ void shade_emissive(MaterialQueue* queue, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, color* framebuffer) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= queue[EMISSIVE].count[0]) return;

    int idx = queue[EMISSIVE].ray_idx[q_idx];
    int p_idx = rays->pixel_idx[idx];

    diffuse_light* light = (diffuse_light*)materials[hits->mat_id[idx]];
    atomicAdd(&framebuffer[p_idx].e[0], throughput[idx].e[0] * light->emitted().e[0]);
    atomicAdd(&framebuffer[p_idx].e[1], throughput[idx].e[1] * light->emitted().e[1]);
    atomicAdd(&framebuffer[p_idx].e[2], throughput[idx].e[2] * light->emitted().e[2]);
    rays->alive[idx] = 0;
}

__global__ void shade_miss(MaterialQueue* queue, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  color* framebuffer) {
    // Background 
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= queue[MISS].count[0]) return;

    int idx = queue[MISS].ray_idx[q_idx];
    int p_idx = rays->pixel_idx[idx];

    
    vec3 unit_direction = unit_vector(vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]));
    float a = 0.5f * (unit_direction.y() + 1.0f);
    color bg = (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    atomicAdd(&framebuffer[p_idx].e[0], throughput[idx].e[0] * bg.e[0]);
    atomicAdd(&framebuffer[p_idx].e[1], throughput[idx].e[1] * bg.e[1]);
    atomicAdd(&framebuffer[p_idx].e[2], throughput[idx].e[2] * bg.e[2]);
    rays->alive[idx] = 0;

}


// ------------------------------------------------------------------------------------------------
// Rendering functions to handle Kernel calls
// ------------------------------------------------------------------------------------------------

cudaError_t launch_init_rng(RngState *d_rng, int image_width, int image_height,
                            uint32_t seed, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((image_width + block.x - 1) / block.x,
            (image_height + block.y - 1) / block.y);
  init_rng_kernel<<<grid, block, 0, stream>>>(d_rng, image_width, image_height,
                                              seed);
  return cudaGetLastError();
}
__host__ void initialize_rays(int samples_per_pixel, int width, int height,
                            camera* cam, color* throughput, rayBuffer* rays, RngState* states){
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x -1)/dimBlock.x, (height + dimBlock.y -1)/dimBlock.y);

    
    init_rays<<<dimGrid, dimBlock>>>(samples_per_pixel, width, height, cam, throughput, rays, states);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void render(
    rayBuffer* rays, hitBuffer* hits, color* throughput, 
    bvh_array* bvh, hittable** objects, material** materials, MaterialQueue* h_queues, MaterialQueue* d_queues,
    color* framebuffer, int n_active, int max_depth, camera* cam, RngState* states){
            
    /* Run the wavefront pathtracing loop */
    dim3 dimBlock(512);
    dim3 dimGrid((n_active + dimBlock.x - 1) / dimBlock.x);
    
    
    for (int bounce = 0; bounce < max_depth; bounce++) {
        // Reset all queues each bounce
        for (int i = 0; i < NUM_MAT_TYPES; i++)
            cudaMemset(h_queues[i].count, 0, sizeof(int));

        intersect<<<dimGrid, dimBlock>>>(hits, rays, bvh, objects, n_active);
        classify<<<dimGrid, dimBlock>>>(hits, rays, d_queues, n_active);
        //shade<<<dimGrid, dimBlock>>>(hits, rays, throughput, framebuffer, materials, n_active, states);
        CUDA_CHECK(cudaDeviceSynchronize());
    

        // Copy counts back

        int h_counts[NUM_MAT_TYPES];
        for (int i = 0; i < NUM_MAT_TYPES; i++)
            CUDA_CHECK(cudaMemcpy(&h_counts[i], h_queues[i].count, sizeof(int), cudaMemcpyDeviceToHost));

        // Launch each shade kernel with its own queue size
        if (h_counts[LAMBERTIAN] > 0){
            dimGrid.x =(h_counts[LAMBERTIAN] + dimBlock.x - 1) / dimBlock.x;
            shade_lambertian<<<dimGrid, dimBlock>>>(d_queues, hits, rays, throughput, materials, states);
        }
        if (h_counts[METAL] > 0){
            dimGrid.x =(h_counts[METAL] + dimBlock.x - 1) / dimBlock.x;
            shade_metal<<<dimGrid, dimBlock>>>(d_queues, hits, rays, throughput, materials, states);
        }
        if (h_counts[DIELECTRIC] > 0){
            dimGrid.x =(h_counts[DIELECTRIC] + dimBlock.x - 1) / dimBlock.x;
            shade_dielectric<<<dimGrid, dimBlock>>>(d_queues, hits, rays, throughput, materials, states);
        }
        if (h_counts[EMISSIVE] > 0){
            dimGrid.x =(h_counts[EMISSIVE] + dimBlock.x - 1) / dimBlock.x;
            shade_emissive<<<dimGrid, dimBlock>>>(d_queues, hits, rays, throughput, materials, framebuffer);
        }
        if (h_counts[MISS] > 0){
            dimGrid.x =(h_counts[MISS] + dimBlock.x - 1) / dimBlock.x;
            shade_miss<<<dimGrid, dimBlock>>>(d_queues, hits, rays, throughput, framebuffer);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update n_active = rays still alive
        //n_active -= h_counts[EMISSIVE] + h_counts[MISS];
        // plus any absorbed rays from scatter returning false
        //if (n_active == 0) break;

        // Update gridsize
        dimGrid.x = (n_active + dimBlock.x - 1) / dimBlock.x ;
    }

}


// ------------------------------------------------------------------------------------------------
// Main function
// ------------------------------------------------------------------------------------------------


int main(){

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);  // 256MB
    // World
    float aspect_ratio = 16.0f/9.0f;
    int num_objects = 488;
    int width = 1200;                             // Image width
    int height = int(float(width)/aspect_ratio);  // Image height
    int samples_per_pixel = 500;
    int max_depth = 4;

    height = (height < 1) ? 1 : height;           // Ensure that image height is at least 1.

    int num_pixels = width * height;  
    


    
    int samples_per_pixels[6] = {450, 550, 600, 650, 700, 750};
    int max_depths[11] = {4, 8, 16, 32}; 
    
    for (int j=0; j<4; j++){
        //samples_per_pixel = samples_per_pixels[j];
        max_depth = max_depths[j];
        int num_rays = num_pixels * samples_per_pixel;
        // Allocate states
        RngState rng;
        rng_seed(rng, base_seed, static_cast<uint32_t>(path))
        RngState* d_states;
        CUDA_CHECK(cudaMalloc(&d_states, num_rays * sizeof(RngState)));
        unsigned long seed = 9999999;

        // Make World
        hittable**  d_list;
        hittable**  d_world;
        material**  d_material;
        color*      d_framebuffer;
        color*      h_framebuffer;
        
        CUDA_CHECK(cudaMalloc(&d_material, num_objects * sizeof(material*)));
        CUDA_CHECK(cudaMalloc(&d_list, num_objects * sizeof(hittable*)));
        CUDA_CHECK(cudaMalloc(&d_world, sizeof(hittable*)));
        CUDA_CHECK(cudaMalloc(&d_framebuffer, num_pixels * sizeof(color)));
        CUDA_CHECK(cudaMallocHost(&h_framebuffer, num_pixels * sizeof(color)));
        CUDA_CHECK(cudaMemset(d_framebuffer, 0, num_pixels * sizeof(color)));
        
        create_complex_world<<<1, 1>>>(d_material, d_list, d_world);
        CUDA_CHECK(cudaDeviceSynchronize());

            
        // Make BVH
        bvh_array* d_bvh;
        int* d_bvh_size;

        CUDA_CHECK(cudaMalloc(&d_bvh, 4*num_objects*sizeof(bvh_array)));
        CUDA_CHECK(cudaMalloc(&d_bvh_size, sizeof(int)));

        build_bvh<<<1,1>>>(d_list, num_objects, d_bvh, d_bvh_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check that BVH build was succesfull
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "build_bvh failed: " << cudaGetErrorString(err) << "\n";


        // Init ray buffer and copy to device
        rayBuffer h_rays;
        CUDA_CHECK(cudaMalloc(&h_rays.ox, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.oy, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.oz, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.dx, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.dy, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.dz, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_rays.alive, num_rays * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_rays.pixel_idx, num_rays * sizeof(int)));

        rayBuffer* d_rays;
        CUDA_CHECK(cudaMalloc(&d_rays, sizeof(rayBuffer)));
        cudaMemcpy(d_rays, &h_rays, sizeof(rayBuffer), cudaMemcpyHostToDevice);
        //std::clog<< "Allocated ray buffers \n";

        // Init hit buffer and copy to device
        hitBuffer h_hits;
        CUDA_CHECK(cudaMalloc(&h_hits.px, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.py, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.pz, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.nx, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.ny, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.nz, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.t, num_rays * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h_hits.mat_id, num_rays * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_hits.mat_type, num_rays * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&h_hits.front_face, num_rays * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&h_hits.hit_anything, num_rays * sizeof(bool)));

        hitBuffer* d_hits;
        CUDA_CHECK(cudaMalloc(&d_hits, sizeof(hitBuffer)));
        CUDA_CHECK(cudaMemcpy(d_hits, &h_hits, sizeof(hitBuffer), cudaMemcpyHostToDevice));
        //std::clog<< "Allocated Hit buffers \n";

        color*  d_throughput;
        CUDA_CHECK(cudaMalloc(&d_throughput, num_rays * sizeof(color)));
        //std::clog<< "Allocated throughput \n";

            // Host side array
        MaterialQueue h_queues[NUM_MAT_TYPES];

        // Allocate device memory for each queue's fields
        for (int i = 0; i < NUM_MAT_TYPES; i++) {
            CUDA_CHECK(cudaMalloc(&h_queues[i].ray_idx, num_rays * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&h_queues[i].count,   sizeof(int)));
            //std::clog << "Allocated queue: " << i << "\n";
        }

        // Copy the struct array (containing device pointers) to device
        MaterialQueue* d_queues;
        CUDA_CHECK(cudaMalloc(&d_queues, NUM_MAT_TYPES * sizeof(MaterialQueue)));
        CUDA_CHECK(cudaMemcpy(d_queues, h_queues, NUM_MAT_TYPES * sizeof(MaterialQueue),
                            cudaMemcpyHostToDevice));


        // Make camera
        camera cam;
        camera* d_cam;

        // Set viewport
        cam.width = width;
        cam.height = height;
        cam.samples_per_pixel = samples_per_pixel;              // Sampling to make smoother edges
        cam.max_depth = max_depth;  
        
        // Control camera position  (Camera for the complex scene)
        cam.vfov     = 20;
        cam.look_from = point3(13,2,3);
        cam.look_at   = point3(0,0,0);
        cam.vup      = vec3(0,1,0);

        cam.defocus_angle = 0.0;
        cam.focus_dist    = 10.0;


        initialize_states(num_rays, seed, d_states);
        //std::clog<< "Initialized random states \n";

        d_cam = cam.move_to_device();
      


        int N = 1;

        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float t = 0;

        
        cudaEventRecord(start, 0);
        for (int i = 0; i<N; i++){
            //std::clog<< "Iteration: " << i+1 << "\n";
            initialize_rays(samples_per_pixel, width, height, d_cam, d_throughput, d_rays, d_states);
            render(d_rays, d_hits, d_throughput, d_bvh, d_list, d_material, h_queues, d_queues, d_framebuffer, num_rays, max_depth, d_cam, d_states);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        std::clog << "\r Average Cuda rendering time (" << N << " runs): " << t/float(N) << "\n";
    

    CUDA_CHECK(cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost));
    
    //std::clog<< "Writting image to png... \n";
    std::string output_name = "complex_image.png";
    // Write to png
    write_png(h_framebuffer, num_pixels, samples_per_pixel, width, height, output_name);
    std::clog<< "Done! \n";

    cudaFree(d_bvh);
    cudaFree(d_framebuffer);
    cudaFree(d_cam);
    cudaFree(d_states);
    cudaFree(d_list); 
    cudaFree(d_world);
    cudaFreeHost(h_framebuffer);

    // Ray buffer fields
    cudaFree(h_rays.ox); cudaFree(h_rays.oy); cudaFree(h_rays.oz);
    cudaFree(h_rays.dx); cudaFree(h_rays.dy); cudaFree(h_rays.dz);
    cudaFree(h_rays.alive); cudaFree(h_rays.pixel_idx);
    cudaFree(d_rays);

    // Hit buffer fields
    cudaFree(h_hits.px); cudaFree(h_hits.py); cudaFree(h_hits.pz);
    cudaFree(h_hits.nx); cudaFree(h_hits.ny); cudaFree(h_hits.nz);
    cudaFree(h_hits.mat_id); cudaFree(h_hits.front_face);
    cudaFree(h_hits.hit_anything);
    cudaFree(d_hits);

    cudaFree(d_throughput);
    cudaFree(d_material);

    // Free all work queues
    for (int i = 0; i < NUM_MAT_TYPES; i++) {
        cudaFree(h_queues[i].ray_idx);
        cudaFree(h_queues[i].count);
    }
    cudaFree(d_queues);
    

    cudaDeviceReset();  // forces printf buffer flush
    }
}
  
