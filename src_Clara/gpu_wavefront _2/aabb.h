#ifndef AABB_h
#define AABB_h

#include "interval.h"


class aabb{
    public:
        interval x, y, z;

       __host__ __device__ aabb () {}; // Default aabb is empty as intervals are empty by default.

       __host__ __device__ aabb(const interval& x, const interval& y, const interval& z): x(x), y(y), z(z) {};

       __host__ __device__ aabb(point3& a, point3& b){
            /* 
                Treat two points as the extrema for the box, 
                so we don't need to rely on a particular order when constructing the box.
            */
            x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
            y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        }
       __host__ __device__ aabb(vec3 a, vec3 b){
            /* 
                Treat two points as the extrema for the box, 
                so we don't need to rely on a particular order when constructing the box.
            */
            x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
            y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
        }

       __host__ __device__ aabb(const aabb& box0, const aabb& box1) {
            x = interval(box0.x, box1.x);
            y = interval(box0.y, box1.y);
            z = interval(box0.z, box1.z);
        }

       __host__ __device__ const interval& axis_interval(int n) const {
            // Return the interval of the n's axis of the aabb
            if (n == 1) return y;
            if (n == 2) return z;
            return x;
        }

       __device__ bool hit(rayBuffer* rays, int idx, interval ray_t) const {
            // Hit detector for BVH

            const point3& ray_orig = point3(rays->ox[idx], rays->oy[idx], rays->oz[idx]);
            const vec3& ray_dir = vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]);

            for (int axis = 0; axis < 3; axis++){
                const interval& ax = axis_interval(axis);
                const float adinv = 1.0 / ray_dir[axis];

                auto t0 = (ax.min - ray_orig[axis]) * adinv;
                auto t1 = (ax.max - ray_orig[axis]) * adinv;

                // Handle if interval is reversed
                if (t0 < t1){
                    if (t0 > ray_t.min) ray_t.min = t0;
                    if (t1 < ray_t.max) ray_t.max = t1;
                } else {
                    if (t1 > ray_t.min) ray_t.min = t1;
                    if (t0 < ray_t.max) ray_t.max = t0;

                }

                if (ray_t.max <= ray_t.min) return false;
            }
            return true;
        }

        __host__ __device__ int longest_axis() const {
            if (x.size() > y.size()) return x.size() > z.size() ? 0:2;
            else return y.size() > z.size() ? 1:2;
        }

        static const aabb empty, universe;
};

__host__ __device__
inline aabb empty_aabb() {
    return aabb(empty_interval(), empty_interval(), empty_interval());
}

__host__ __device__
inline aabb universe_aabb() {
    return aabb(universe_interval(), universe_interval(), universe_interval());
}

#endif