#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "vec3.h"




class material{
    public:
        int mat_id;
        MatType type;
        __device__ material(int id, MatType type) : mat_id(id), type(type) {}
        __device__ virtual ~material() = default;

        __device__ virtual bool scatter(hitBuffer* hits, rayBuffer* rays, int idx, color& attenuation, curandStatePhilox4_32_10_t* state) const {
            return false;
        }
};

class lambertian : public material {
    public:
        color albedo;
        __device__ lambertian(int id, const color& albedo) : material(id, LAMBERTIAN), albedo(albedo) {}

        __device__ bool scatter(hitBuffer* hits, rayBuffer* rays, int idx, color& attenuation, curandStatePhilox4_32_10_t* state) const override {
            
            vec3 normal = vec3(hits->nx[idx], hits->ny[idx],  hits->nz[idx]);
            auto scatter_direction = normal + random_unit_vector(state);

            // Catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = normal;

            store_ray(point3(hits->px[idx], hits->py[idx],  hits->pz[idx]), scatter_direction, rays, idx);
            attenuation = albedo;
            return true;
        }
        
};

class metal : public material {
    public: 
        color albedo;
        float fuzz;
        __device__ metal(int id, const color& albedo, float fuzz) : material(id, METAL), albedo(albedo), fuzz(fuzz < 1 ? fuzz: 1) {}

        __device__ bool scatter(hitBuffer* hits, rayBuffer* rays, int idx, color& attenuation, curandStatePhilox4_32_10_t* state) const override {
            

            vec3 normal = vec3(hits->nx[idx], hits->ny[idx], hits->nz[idx]);
            vec3 reflected = reflect(vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]), normal);
            reflected  = unit_vector(reflected) + (fuzz * random_unit_vector(state));

            store_ray(point3(hits->px[idx], hits->py[idx],  hits->pz[idx]), reflected, rays, idx); // Scattered ray with direction = reflected
            attenuation = albedo;
            return (dot(reflected, normal) > 0); 
        }

};

class dielectric : public material {
    /* Refractive materials such as glass or water */
    public:
        __host__ __device__ dielectric(int id, float refraction_index) : material(id, DIELECTRIC), refraction_index(refraction_index) {}

        __device__ bool scatter(hitBuffer* hits, rayBuffer* rays, int idx, color& attenuation, curandStatePhilox4_32_10_t* state) const override {
            attenuation = color(1.0f, 1.0f, 1.0f);
            float ri = hits->front_face[idx] ? (1.0f /refraction_index) : refraction_index;

            vec3 normal = vec3(hits->nx[idx], hits->ny[idx],  hits->nz[idx]);

            vec3 unit_direction = unit_vector(vec3(rays->dx[idx], rays->dy[idx], rays->dz[idx]));
            float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
            float sin_theta = sqrtf(1.0 -cos_theta * cos_theta);

            bool cannot_refrac = sin_theta * ri > 1.0;
            vec3 direction;
            
            if (cannot_refrac || reflectance(cos_theta, ri) > random_float(state))
                direction = reflect(unit_direction, normal);
            else
                direction = refract(unit_direction, normal, ri);

            store_ray(point3(hits->px[idx], hits->py[idx],  hits->pz[idx]), direction, rays, idx);
            return true;
        }

    private:
        // Refractive index in vacuum or air, or the ratio of the material's refractive index over
        // the refractive index of the enclosing media
        float refraction_index;

        __device__ static float reflectance(float cosine, float refraction_index){
            // Uses Schlicks approximation for reflectance

            auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
            r0 = r0 * r0;

            return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
        }
};

class diffuse_light : public material {
public:
    color emit;
    __device__ diffuse_light(int id, const color& emit) : material(id, EMISSIVE), emit(emit) {}
    
    __device__ bool scatter(hitBuffer* hit_buf, rayBuffer* r_buf, int h_idx,
                            color& attenuation, curandStatePhilox4_32_10_t* state) const override {
        return false;  // never scatters
    }

    __device__ color emitted() const { return emit; }
    
};

struct MaterialQueue {
    int* ray_idx;    // which rays hit this material
    int* count;      // atomic counter for queue size
};



#endif