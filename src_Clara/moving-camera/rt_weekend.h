#ifndef RT_WEEKEND_H
#define RT_WEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>
#include <omp.h>
#include <curand_kernel.h>
#include <string>
//#include <cuda_runtime.h>


// Constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;

//Utility functions
__host__ __device__ inline float degrees_to_radians(float degrees){
    return degrees * pi / 180.0;
}

#ifdef __CUDACC__
__global__ void initRandKernel(curandState* states, int width, int height, unsigned long seed); 
#endif

__device__ inline float random_float(curandState* state) {
    return float(curand_uniform_double(state));  // returns (0, 1]
}


__device__ inline float random_float(float min, float max, curandState* state){
    // Return random uniformely distributed float in interval [min, max)
    return min + (max - min) * random_float(state);
}

// Simple LCG generator for use on 1 thread

__device__ inline float lcg_random(uint32_t& state) {
    state = 1664525u * state + 101394223u;
    return state / float(0xFFFFFFFFu);
}

__device__ inline float lcg_random(float min, float max, uint32_t& state) {
    return min + (max - min) * lcg_random(state);
}


// Headers
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "interval.h"


#endif