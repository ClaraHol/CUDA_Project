// raytracer.cuh
// Declaration of the kernel launch wrapper called from main.cpp.

#pragma once
#include <cuda_runtime.h>
#include "rt_weekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"



// frame — frame counter (useful for animations / progressive refinement)

void initRand(curandState* states, int width, int height, unsigned long seed);
void initWorld(bvh_array* d_bvh, hittable** d_list, hittable** d_world, int* d_bvh_size, 
                int num_objects, int width, int height);

void initCamera(camera& h_cam, camera*& d_cam, int image_height, int image_width, int samples_pp, int max_depth, float vfov, 
                point3 look_from, point3 look_at, point3 vup, float defocus_angle, float focus_dist);

void launchRayTracer( bvh_array* bvh_flat, hittable** d_list, uchar4* pbo, int width, int height,
                    int samples_per_pixel, int max_depth, int frame, camera* cam, curandState* states,
                    cudaStream_t stream);
void launchAccumRayTracer(bvh_array* bvh_flat, hittable** d_list, float3* accum, uchar4* pbo, int width, int height,
                    int max_depth, int samples_per_pixel, int frame, camera* cam, curandState* states, cudaStream_t stream);

