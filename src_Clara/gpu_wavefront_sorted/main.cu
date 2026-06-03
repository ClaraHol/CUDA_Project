
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

// Quick debug kernel
__global__ void count_alive(rayBuffer* rays, int* total, int n_rays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_rays) return;
    if (rays->alive[idx]) atomicAdd(total, 1);
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

__global__ void init_rays(int samples_per_pixel, int width, int height, camera* cam, color* throughput, rayBuffer* rays, curandState* state){
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
    hits->hit_anything[idx] = 0;

    // Run the hit test (Handles all the writing to hits and rays in the sphere hit function)
    bvh[0].hit(bvh, objects, interval(0.001, infinity), hits, rays, idx);
} 

__global__ void count_materials(hitBuffer* hits, rayBuffer* rays,
                          int* counts, int n_rays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_rays) return;
    if (!rays->alive[idx]) return;

    int mat_type = hits->hit_anything[idx] ? hits->mat_type[idx] : MISS;
    
    // Add one to the material count
    atomicAdd(&counts[mat_type], 1);
}

__global__ void counting_sort(hitBuffer* hits, rayBuffer* rays,
                               int* sorted_indices, int* offsets, 
                               int* counts, int n_rays) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_rays) return;
    if (!rays->alive[idx]) return;

    int mat_type = hits->hit_anything[idx] ? hits->mat_type[idx] : MISS;
    
    // Debug -- check mat_type is valid
    if (mat_type < 0 || mat_type >= NUM_MAT_TYPES) {
        printf("invalid mat_type %d at idx %d\n", mat_type, idx);
        return;
    }
    
    // Atomically claim a slot in the output
    int pos = atomicAdd(&counts[mat_type], 1);

    // Debug -- check write position is valid
    if (pos >= n_rays) {
        printf("out of bounds write: mat_type=%d pos=%d offset=%d write_pos=%d\n",
               mat_type, pos, offsets[mat_type], pos);
        return;
    }
    sorted_indices[offsets[mat_type] + pos] = idx;
}

__global__ __launch_bounds__(384, 7) void shade_lambertian(int* ray_indices, int n, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, curandState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= n) return;
    int num_rays = 81000000;


    int idx = ray_indices[q_idx];
    if (idx < 0 || idx >= num_rays) return;  // invalid index from previous bounce
    lambertian* mat = (lambertian*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {

        rays->alive[idx] = 0;
    }
}


__global__ __launch_bounds__(384, 7) void shade_metal(int* ray_indices, int n, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, curandState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= n) return;
    int num_rays = 81000000;

    int idx = ray_indices[q_idx];
    if (idx < 0 || idx >= num_rays) return;  // invalid index from previous bounce
    metal* mat = (metal*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {
        rays->alive[idx] = 0;
    }

}

__global__ __launch_bounds__(384, 7) void shade_dielectric(int* ray_indices, int n, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, curandState* states) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= n) return;

    int idx = ray_indices[q_idx];
    int num_rays = 81000000;

    if (idx < 0 || idx >= num_rays) return;  // invalid index from previous bounce
    dielectric* mat = (dielectric*)materials[hits->mat_id[idx]];

    color attenuation;
    if (mat->scatter(hits, rays, idx, attenuation, &states[idx])) {
        throughput[idx] *= attenuation;
    } else {
        rays->alive[idx] = 0;
    }
}

__global__ __launch_bounds__(384, 7) void shade_emissive(int* ray_indices, int n, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  material** materials, color* framebuffer) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= n) return;

    int idx = ray_indices[q_idx];
    int num_rays = 81000000;

    if (idx < 0 || idx >= num_rays) return;  // invalid index from previous bounce
    int p_idx = rays->pixel_idx[idx];

    diffuse_light* light = (diffuse_light*)materials[hits->mat_id[idx]];
    atomicAdd(&framebuffer[p_idx].e[0], throughput[idx].e[0] * light->emitted().e[0]);
    atomicAdd(&framebuffer[p_idx].e[1], throughput[idx].e[1] * light->emitted().e[1]);
    atomicAdd(&framebuffer[p_idx].e[2], throughput[idx].e[2] * light->emitted().e[2]);
    rays->alive[idx] = 0;
}

__global__ void shade_miss(int* ray_indices, int n, hitBuffer* hits,
                                  rayBuffer* rays, color* throughput,
                                  color* framebuffer) {
    // Background 
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= n) return;

    int idx = ray_indices[q_idx];
    int num_rays = 81000000;

    if (idx < 0 || idx >= num_rays) return;  // invalid index from previous bounce
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

__host__ void initialize_states(int num_rays, unsigned long seed, curandState* states){
    /* Run the wavefront pathtracing loop */
    dim3 dimBlock(512);
    dim3 dimGrid((num_rays + dimBlock.x - 1) / dimBlock.x);

    // Init random seeds for each ray
    init_rand_ray<<<dimGrid, dimBlock>>>(num_rays, seed, states);
    CUDA_CHECK(cudaDeviceSynchronize());

}

__host__ void initialize_rays(int samples_per_pixel, int width, int height,
                            camera* cam, color* throughput, rayBuffer* rays, curandState* states){
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x -1)/dimBlock.x, (height + dimBlock.y -1)/dimBlock.y);

    
    init_rays<<<dimGrid, dimBlock>>>(samples_per_pixel, width, height, cam, throughput, rays, states);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__host__ void render(
    rayBuffer* rays, hitBuffer* hits, color* throughput, 
    bvh_array* bvh, hittable** objects, material** materials, int* d_counts, int* d_offsets, int* d_sorted_indices,
    color* framebuffer, int n_active, int max_depth, camera* cam, curandState* states){
            
    /* Run the wavefront pathtracing loop */
    dim3 dimBlock(384);
    dim3 dimGrid((n_active + dimBlock.x - 1) / dimBlock.x);
    cudaStream_t stream0, stream1, stream2, stream3, stream4;
    
    for (int bounce = 0; bounce < max_depth; bounce++) {
        // Reset
        cudaMemset(d_sorted_indices, -1, n_active * sizeof(int));
        cudaMemset(d_counts, 0, NUM_MAT_TYPES * sizeof(int));

        intersect<<<dimGrid, dimBlock>>>(hits, rays, bvh, objects, n_active);
        
        // First pass -- count rays per material (same as your classify)
        count_materials<<<dimGrid, dimBlock>>>(hits, rays, d_counts, n_active);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Compute offsets on host
        int h_counts[NUM_MAT_TYPES];
        cudaMemcpy(h_counts, d_counts, NUM_MAT_TYPES * sizeof(int), cudaMemcpyDeviceToHost);
        
        int h_offsets[NUM_MAT_TYPES];
        h_offsets[0] = 0;
        for (int i = 1; i < NUM_MAT_TYPES; i++)
            h_offsets[i] = h_offsets[i-1] + h_counts[i-1];
        
        cudaMemcpy(d_offsets, h_offsets, NUM_MAT_TYPES * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_counts, 0, NUM_MAT_TYPES * sizeof(int));  // reset for second pass

        // Second pass -- scatter rays into sorted positions
        counting_sort<<<dimGrid, dimBlock>>>(hits, rays, d_sorted_indices, d_offsets, d_counts, n_active);
        CUDA_CHECK(cudaDeviceSynchronize());
    

        // Launch each shade kernel with its own queue size
        if (h_counts[LAMBERTIAN] > 0){
            dimGrid.x =(h_counts[LAMBERTIAN] + dimBlock.x - 1) / dimBlock.x;
            shade_lambertian<<<dimGrid, dimBlock, 0, stream0>>>(d_sorted_indices + h_offsets[LAMBERTIAN], h_counts[LAMBERTIAN], hits, rays, throughput, materials, states);
        }
        if (h_counts[METAL] > 0){
            dimGrid.x =(h_counts[METAL] + dimBlock.x - 1) / dimBlock.x;
            shade_metal<<<dimGrid, dimBlock, 0, stream1>>>(d_sorted_indices + h_offsets[METAL], h_counts[METAL], hits, rays, throughput, materials, states);
        }
        if (h_counts[DIELECTRIC] > 0){
            dimGrid.x =(h_counts[DIELECTRIC] + dimBlock.x - 1) / dimBlock.x;
            shade_dielectric<<<dimGrid, dimBlock, 0, stream2>>>(d_sorted_indices + h_offsets[DIELECTRIC], h_counts[DIELECTRIC], hits, rays, throughput, materials, states);
        }
        if (h_counts[EMISSIVE] > 0){
            dimGrid.x =(h_counts[EMISSIVE] + dimBlock.x - 1) / dimBlock.x;
            shade_emissive<<<dimGrid, dimBlock, 0, stream3>>>(d_sorted_indices + h_offsets[EMISSIVE], h_counts[EMISSIVE], hits, rays, throughput, materials, framebuffer);
        }
        if (h_counts[MISS] > 0){
            dimGrid.x =(h_counts[MISS] + dimBlock.x - 1) / dimBlock.x;
            shade_miss<<<dimGrid, dimBlock, 0, stream4>>>(d_sorted_indices + h_offsets[MISS], h_counts[MISS], hits, rays, throughput, framebuffer);
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
    int samples_per_pixel = 100;
    int max_depth = 10;

    height = (height < 1) ? 1 : height;           // Ensure that image height is at least 1.

    int num_pixels = width * height;  
    int num_rays = num_pixels * samples_per_pixel;


    // Allocate states
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, num_rays * sizeof(curandState)));
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
    std::clog<< "Allocated ray buffers \n";

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
    std::clog<< "Allocated Hit buffers \n";

    color*  d_throughput;
    CUDA_CHECK(cudaMalloc(&d_throughput, num_rays * sizeof(color)));
    std::clog<< "Allocated throughput \n";
    std::clog<< "NUM_MAT_TYPES: " << NUM_MAT_TYPES << " \n";

    // One sorted index array instead of per-material queues
    int* d_sorted_indices;
    cudaMalloc(&d_sorted_indices, num_rays * sizeof(int));

    // Offsets and counts on device
    int* d_offsets;
    int* d_counts;
    cudaMalloc(&d_offsets, NUM_MAT_TYPES * sizeof(int));
    cudaMalloc(&d_counts,  NUM_MAT_TYPES * sizeof(int));


    // Make camera
    camera cam;

    // Set viewport
    cam.width = width;
    cam.height = height;
    cam.samples_per_pixel = samples_per_pixel;              // Sampling to make smoother edges
    cam.max_depth = max_depth;                              // Maximum recursion depth
    
    // Control camera position  (Camera for the complex scene)
    cam.vfov     = 20;
    cam.look_from = point3(13,2,3);
    cam.look_at   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 0.0;
    cam.focus_dist    = 10.0;

    camera* d_cam = cam.move_to_device();


    int N = 10;
    

    initialize_states(num_rays, seed, d_states);
    std::clog<< "Initialized random states \n";

    auto t = omp_get_wtime();
    

    for (int i = 0; i<N; i++){
        std::clog<< "Iteration: " << i+1 << "\n";
        initialize_rays(samples_per_pixel, width, height, d_cam, d_throughput, d_rays, d_states);
        render(d_rays, d_hits, d_throughput, d_bvh, d_list, d_material, d_counts, d_offsets, d_sorted_indices, d_framebuffer, num_rays, max_depth, d_cam, d_states);
    }
    t = omp_get_wtime() - t;
    std::clog << "\r Average Cuda rendering time (" << N << " runs): " << t/float(N) << "\n";

    CUDA_CHECK(cudaMemcpy(h_framebuffer, d_framebuffer, num_pixels * sizeof(color), cudaMemcpyDeviceToHost));
    
    std::clog<< "Writting image to png... \n";
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
    cudaFree(d_sorted_indices);
    cudaFree(d_offsets);
    cudaFree(d_counts);

    cudaDeviceReset();  // forces printf buffer flush
}
  
