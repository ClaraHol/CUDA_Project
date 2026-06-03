#ifndef RAY_H
#define RAY_H

#include "vec3.h"
#include "cuda_compat.h"

/*
class ray{
    public:
        point3 orig;
        vec3 dir;

        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

        __device__ const point3& origin() const { return orig; }
        __device__ const vec3& direction() const {return dir; }

        __device__ point3 at(float t) const{
            return orig + t*dir;
        }

};
*/

class rayBuffer{
    public:
        float* ox; float* oy; float* oz;    // Origin coordinates
        float* dx; float* dy; float* dz;    // Direction coodinates
        int* alive;                         // Mark if ray has terminated
        int* pixel_idx;                     // Corresponding pixel idx

        
};

/*

__device__ inline ray load_ray(const rayBuffer* rays, int idx){
    ray r; 
    r.orig = point3(rays->ox[idx], rays->oy[idx], rays->oz[idx]);
    r.dir = vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]);
    return r;
};
*/
__device__ inline void init_ray(const point3& orig, const vec3& dir, int is_alive, int p_idx, rayBuffer* rays, int idx){
    // Save origin
    rays->ox[idx] = orig.x();
    rays->oy[idx] = orig.y();
    rays->oz[idx] = orig.z();
    
    //Save direction
    rays->dx[idx] = dir.x();
    rays->dy[idx] = dir.y();
    rays->dz[idx] = dir.z();

    // Mark if ray is still alive and pixel index
    rays->alive[idx] = is_alive;
    rays->pixel_idx[idx] = p_idx;

};

__device__ inline void store_ray(const point3& orig, const vec3& dir, rayBuffer* rays, int idx){
    // Save origin
    rays->ox[idx] = orig.x();
    rays->oy[idx] = orig.y();
    rays->oz[idx] = orig.z();
    
    //Save direction
    rays->dx[idx] = dir.x();
    rays->dy[idx] = dir.y();
    rays->dz[idx] = dir.z();

};

#endif