#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "vec3.h"

class material{
    public:
        __device__ virtual ~material() = default;

        __device__ virtual bool scatter( const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const {
            return false;
        }
};

class lambertian : public material {
    public:
        __device__ lambertian(const color& albedo) : albedo(albedo) {}

        __device__ bool scatter( const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const override {
            auto scatter_direction = rec.normal + random_unit_vector(state);

            // Catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }

    private:
        color albedo;
};

class metal : public material {
    public: 
        __device__ metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz: 1) {}

        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const override {

            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected  = unit_vector(reflected) + (fuzz * random_unit_vector(state));

            scattered = ray(rec.p, reflected);
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    private:
        color albedo;
        float fuzz;
};

class dielectric : public material {
    /* Refractive materials such as glass or water */
    public:
        __host__ __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state) const override {
            attenuation = color(1.0, 1.0, 1.0);
            float ri = rec.front_face ? (1.0 /refraction_index) : refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
            float sin_theta = sqrtf(1.0 -cos_theta * cos_theta);

            bool cannot_refrac = sin_theta * ri > 1.0;
            vec3 direction;

            if (cannot_refrac || reflectance(cos_theta, ri) > random_double(state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, ri);

            scattered = ray(rec.p, direction);
            return true;
        }

    private:
        // Refractive index in vacuum or air, or the ratio of the material's refractive index over
        // the refractive index of the enclosing media
        float refraction_index;

        __device__ static float reflectance(float cosine, float refraction_index){
            // Uses Schlicks approximation for reflectance

            auto r0 = (1 - refraction_index) / (1 + refraction_index);
            r0 = r0 * r0;

            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }
};



#endif