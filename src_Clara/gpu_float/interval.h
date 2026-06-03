#ifndef INTERVAL_H
#define INTERVAL_H

#include "rt_weekend.h"

class interval{
    public:
        float min, max;

        __host__ __device__ interval() : min(infinity), max(-infinity) {} // Default interval is empty

        __host__ __device__ interval(float min, float max): min(min), max(max) {}

        __host__ __device__ float size() const {
            return max - min;
        }

        __host__ __device__ bool contains(float x) {
            return min <= x && x <= max;
        }

        __host__ __device__ bool surrounds(float x){
            return min < x && x < max;
        }

        __host__ __device__ float clamp(float x) const{
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        __host__ __device__ interval expand(float delta) const {
            float padding = delta / 2.0;
            return interval(min -padding, max + padding);
        }
        
        __host__ __device__ interval(const interval& a, const interval& b) {
            // Create the interval tightly enclosing the two input intervals.
            min = a.min <= b.min ? a.min : b.min;
            max = a.max >= b.max ? a.max : b.max;
        }

        static const interval empty, universe;
         

};

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity); 

#endif