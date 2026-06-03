#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_weekend.h"
#include "hittable.h"
#include "aabb.h"
#include "bvh.h"

#include <vector>


class hittable_list : public hittable {
    public:
        hittable** objects;
        int size;

        __device__ hittable_list(hittable** objects, int size) : objects(objects), size(size) {}
        __device__ virtual ~hittable_list() = default;

        __device__ bool hit(hitBuffer* hits, rayBuffer* rays, int idx, interval ray_t) const override {
            
            float closest_so_far = ray_t.max;
            bool hit = false;

            for (int i = 0; i < size; i++) {
                // If the object is hit return true and update the record and set it to be the closest object
                if (objects[i] -> hit(hits, rays, idx, interval(ray_t.min, closest_so_far))) {
                    hit = true;
                    closest_so_far = hits->t[idx];
                }
            }
            hits->hit_anything[idx] = hit;  
            return hit;  
        }

        __device__ aabb bounding_box() const override {return bbox;}
    
    private:
        aabb bbox;

};

#endif