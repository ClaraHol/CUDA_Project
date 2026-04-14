#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rt_weekend.h"
#include "hittable.h"

#include <vector>


class hittable_list : public hittable {
    public:
        hittable** objects;
        int size;

        __device__ hittable_list(hittable** objects, int size) : objects(objects), size(size) {}
        __device__ virtual ~hittable_list() = default;

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            float closest_so_far = ray_t.max;

            for (int i = 0; i < size; i++) {
                // If the object is hit return true and update the record and set it to be the closest object
                if (objects[i] -> hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }  
            return hit_anything;  
        }
};

#endif