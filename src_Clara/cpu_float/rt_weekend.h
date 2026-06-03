#ifndef RT_WEEKEND_H
#define RT_WEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

// C++ Std usings
using std::make_shared;
using std::shared_ptr;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const float pi = 3.1415926535897932385;

//Utility functions
inline float degrees_to_radians(float degrees){
    return degrees * pi / 180.0;
}

inline float random_float(){
    // Return random uniformely distributed float in interval [0, 1) thread safe version
    thread_local uint32_t state = 123456789 + omp_get_thread_num();
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state & 0xFFFFFF) / float(0x1000000);
}

inline float random_float(float min, float max){
    // Return random uniformely distributed float in interval [min, max)
    return min + (max - min)*random_float();
}


// Headers
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "interval.h"

#endif