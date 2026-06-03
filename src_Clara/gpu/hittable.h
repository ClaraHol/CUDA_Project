#ifndef HITTABLE_H
#define HITTABLE_H

#include "rt_weekend.h"

class material;

class hit_record {
    public:
        point3 p;
        vec3 normal;
        material* mat;
        double t;
        bool front_face;

        __device__ void set_face_norm(const ray& r, const vec3& outward_normal){
            // Sets the hit record norm
            // Note that outward_normal is assumed to be a unit vector

            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
             
        }
};

class hittable {
    public:
        __device__ virtual ~hittable() = default;
        __device__ virtual bool hit(const ray& r,  interval ray_t, hit_record& rec) const = 0;
};

#endif
