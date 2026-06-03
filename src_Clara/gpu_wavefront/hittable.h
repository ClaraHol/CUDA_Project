#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_weekend.h"
#include "aabb.h"
#include "bvh.h"

enum MatType { LAMBERTIAN, METAL, DIELECTRIC, EMISSIVE, MISS, NUM_MAT_TYPES };
class material;


class hitBuffer {
    public:
        float* px; float* py; float* pz;    // Point coordinates
        float* nx; float* ny; float* nz;    // Normal coordinates
        int* mat_id;                        // Material id to look up in material array
        int* mat_type;                  // Material type for generating queues
        float* t;                           // Time point
        bool* front_face;                   // Front face bool
        bool*  hit_anything; 

        __device__ void set_face_norm(rayBuffer* rays, int idx, const vec3& outward_normal) {
            

            front_face[idx] = dot(vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]), outward_normal) < 0;
            auto normal = front_face[idx] ? outward_normal : -outward_normal;

            nx[idx] = normal.x();
            ny[idx] = normal.y();
            nz[idx] = normal.z();
        }
};


class hittable {
    public:
        __device__ virtual ~hittable() = default;
        __device__ virtual bool hit(hitBuffer* hits, rayBuffer* rays, int idx, interval ray_t) const = 0;
        __device__ virtual aabb bounding_box() const = 0;
};


#endif
