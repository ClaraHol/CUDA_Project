#pragma once

#include <cuda_runtime.h>
#include <cmath>

inline __host__ __device__ float3 make_vec3(float x, float y, float z)
{
    return make_float3(x, y, z);
}

inline __host__ __device__ float3 add3(const float3 &a, const float3 &b)
{
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 sub3(const float3 &a, const float3 &b)
{
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 mul3(const float3 &a, float t)
{
    return make_vec3(a.x * t, a.y * t, a.z * t);
}

inline __host__ __device__ float3 mul3(const float3 &a, const float3 &b)
{
    return make_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 div3(const float3 &a, float t)
{
    return mul3(a, 1.0f / t);
}

inline __host__ __device__ float dot3(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross3(const float3 &a, const float3 &b)
{
    return make_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline __host__ __device__ float len_sq3(const float3 &v)
{
    return dot3(v, v);
}

inline __host__ __device__ float len3(const float3 &v)
{
    return sqrtf(len_sq3(v));
}

inline __host__ __device__ float3 unit3(const float3 &v)
{
    return div3(v, len3(v));
}

inline __host__ __device__ float clampf(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

inline __host__ __device__ float3 reflect3(const float3 &v, const float3 &n)
{
    return sub3(v, mul3(n, 2.0f * dot3(v, n)));
}

inline __host__ __device__ float3 refract3(const float3 &uv, const float3 &n, float eta_ratio)
{
    float cos_theta = fminf(dot3(mul3(uv, -1.0f), n), 1.0f);
    float3 r_out_perp = mul3(add3(uv, mul3(n, cos_theta)), eta_ratio);
    float3 r_out_parallel = mul3(n, -sqrtf(fabsf(1.0f - len_sq3(r_out_perp))));
    return add3(r_out_perp, r_out_parallel);
}

enum MaterialType : int
{
    MAT_LAMBERTIAN = 0,
    MAT_METAL = 1,
    MAT_DIELECTRIC = 2,
};

struct GpuMaterial
{
    MaterialType type;
    float3 albedo;
    float fuzz;
    float ref_idx;
};

struct GpuSphere
{
    float3 center;
    float radius;
    int material_index;
};

struct GpuScene
{
    const GpuSphere *spheres;
    int sphere_count;
    const GpuMaterial *materials;
    int material_count;
};

struct Ray
{
    float3 origin;
    float3 dir;
};

inline __host__ __device__ float3 ray_at(const Ray &r, float t)
{
    return add3(r.origin, mul3(r.dir, t));
}

struct Hit
{
    float3 p;
    float3 normal;
    float t;
    bool front_face;
    int material_index;
};

inline __host__ __device__ void set_face_normal(Hit &h, const Ray &r, const float3 &outward_normal)
{
    h.front_face = dot3(r.dir, outward_normal) < 0.0f;
    h.normal = h.front_face ? outward_normal : mul3(outward_normal, -1.0f);
}
