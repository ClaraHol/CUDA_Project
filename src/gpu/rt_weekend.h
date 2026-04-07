#ifndef RT_WEEKEND_H
#define RT_WEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>
#include <omp.h>
#include <curand_kernel.h>
#include "cuda_compat.h"
//#include <cuda_runtime.h>


// Constants
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

//Utility functions
__host__ __device__ inline double degrees_to_radians(double degrees){
    return degrees * pi / 180.0;
}


__global__ void init_rand(curandState* states, int width, int height, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= width || j >= height) return;

    int idx = i + j * width;

    // Each thread/ray gets a unique sequence to avoid correlation
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline double random_double(curandState* state) {
    return curand_uniform_double(state);  // returns (0, 1]
}


__device__ inline double random_double(double min, double max, curandState* state){
    // Return random uniformely distributed double in interval [min, max)
    return min + (max - min) * random_double(state);
}


// Headers
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "interval.h"


// Check for cuda errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            exit(1); \
        } \
    } while(0)

#endif