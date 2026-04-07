#ifndef VEC3_H
#define VEC3_H

#include "rt_weekend.h"

class vec3 {
    /* Vector class that can be used both on host and device*/
    public:
        double e[3];
        __host__ __device__ vec3() : e{0,0,0}{}
        __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        __host__ __device__ double x() const { return e[0]; }
        __host__ __device__ double y() const { return e[1]; }
        __host__ __device__ double z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ double operator[](int i) const {return e[i]; }
        __host__ __device__ double& operator[] (int i) {return e[i]; }

        __host__ __device__ vec3& operator+=(const vec3& v){
            e[0]+=v.e[0];
            e[1]+=v.e[1];
            e[2]+=v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator*=(double t){
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;

            return *this;
        }

        __host__ __device__ vec3& operator/=(double t){
            return *this *= 1/t;
        }

        __host__ __device__ double length() const{
            return sqrt(length_squared());
        }

        __host__ __device__ double length_squared() const{
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        __host__ __device__ bool near_zero() const {
            // Return true if vector is almost the zero vector
            auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        __device__ static vec3 random(curandState* state){
            return vec3(random_double(state), random_double(state), random_double(state));
        }

        __device__ static vec3 random(double min, double max, curandState* state){
            return vec3(random_double(min, max, state), random_double(min, max, state), random_double(min, max, state));
        }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v){
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v){
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v){
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v){
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v){
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t){
    return t*v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t){
    return (1/t)*v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v){
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v){
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v){
    return v/v.length();
}

__device__ inline vec3 random_unit_vector(curandState* state){
    while (true){
        auto p = vec3::random(-1, 1, state);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1) // Avoid interger undeflow to zero
            return p/sqrt(lensq); 
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* state){
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0) return on_unit_sphere;
    return -on_unit_sphere;
}

__device__ inline vec3 random_on_unit_disk(curandState* state){
    /* 
    Sample random unit vector on disk by sampling polar coordinates and transforming them to cartesian.
    This avoids the while loop of the previous implementation which leads to thread divergence.
    */

    auto r = sqrt(random_double(state));         // Sample radius
    auto a = random_double(0.0, 2*pi, state);    // Sample angle

    vec3 p = vec3(r*cos(a), r*sin(a), 0.0);     // Convert to polar coordinates
    return p;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n){
    return v - 2*dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, double ratio){
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = ratio * (uv + cos_theta *n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n; 

    return r_out_parallel + r_out_perp;
}

#endif