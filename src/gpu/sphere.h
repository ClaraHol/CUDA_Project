#ifndef SPHERE_H
#define SPHERE_H

#include "rt_weekend.h"
#include "hittable.h"


class sphere : public hittable {
    public:
        __device__ sphere(const point3& center, double radius, material* mat) : center(center), radius(radius) , mat(mat) {}


        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            vec3 oc = center - r.origin();
            auto a = r.direction().length_squared();
            auto h = dot(r.direction(), oc);
            auto c = oc.length_squared() - radius * radius;

            auto disc = h*h - a*c;

            if (disc < 0)
                return false;

            auto sqrtd = sqrt(disc);

            // Find nearest root in acceptable distance

            auto root = (h - sqrtd) / a;
            if (!ray_t.surrounds(root)){
                root = (h + sqrtd) / a;
                if (!ray_t.surrounds(root))
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            rec.mat = mat;

            vec3 outward_normal = (rec.p - center)/radius;
            rec.set_face_norm(r, outward_normal);

            return true;
        }
    private:
        point3 center;
        double radius;
        material* mat;

};

#endif