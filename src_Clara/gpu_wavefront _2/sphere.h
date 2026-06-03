#ifndef SPHERE_H
#define SPHERE_H

#include "rt_weekend.h"
#include "hittable.h"



class sphere : public hittable {
    public:
        __device__ sphere(const point3& center, float radius, material* mat) : center(center), radius(radius) , mat(mat) {
            vec3 rvec = vec3(radius, radius, radius);
            bbox = aabb(center - rvec, center + rvec);
        }


        __device__ bool hit(hitBuffer* hits, rayBuffer* rays, int idx, interval ray_t) const override{
            
            vec3 direction = vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]);
            point3 origin = point3(rays->ox[idx], rays->oy[idx], rays->oz[idx]);
            vec3 oc = center - origin;
            auto a = direction.length_squared();
            auto h = dot(direction, oc);
            auto c = oc.length_squared() - radius * radius;

            auto disc = h*h - a*c;

            if (disc < 0)
                return false;

            auto sqrtd = sqrtf(disc);

            // Find nearest root in acceptable distance

            auto root = (h - sqrtd) / a;
            if (!ray_t.surrounds(root)){
                root = (h + sqrtd) / a;
                if (!ray_t.surrounds(root))
                    return false;
            }
            auto p = origin + root * direction;
            hits->t[idx] = root;
            hits->px[idx] = p.x();
            hits->py[idx] = p.y();
            hits->pz[idx] = p.z();
            hits->mat_id[idx] = mat->mat_id;
            hits->mat_type[idx] = (int)mat->type;
            hits->hit_anything[idx] = true;
           

            vec3 outward_normal = (p - center)/radius;
            hits->set_face_norm(rays, idx,  outward_normal);
            

            return true;
        }

        __device__ aabb bounding_box() const override {return bbox;}
    private:
        point3 center;
        float radius;
        material* mat;
        aabb bbox;

};

#endif