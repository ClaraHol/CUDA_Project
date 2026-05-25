#include "bvh.cuh"
#include "kernels.cuh"

#include <math.h>

namespace {

struct PathState {
  Ray ray;
  float3 throughput;
  float3 radiance;
  RngState rng;
  int pixel_index;
  int depth;
  int active;
};

struct PathHit {
  Hit rec;
  int hit;
};

inline __device__ float3 random_in_unit_sphere(RngState &rng) {
  while (true) {
    float3 p = make_vec3(2.0f * rng_next_f32(rng) - 1.0f,
                         2.0f * rng_next_f32(rng) - 1.0f,
                         2.0f * rng_next_f32(rng) - 1.0f);
    float lsq = len_sq3(p);
    if (lsq > 1e-8f && lsq <= 1.0f) {
      return p;
    }
  }
}

inline __device__ float3 random_unit_vector(RngState &rng) {
  return unit3(random_in_unit_sphere(rng));
}

inline __device__ float3 random_in_unit_disk(RngState &rng) {
  while (true) {
    float3 p = make_vec3(2.0f * rng_next_f32(rng) - 1.0f,
                         2.0f * rng_next_f32(rng) - 1.0f, 0.0f);
    if (len_sq3(p) < 1.0f) {
      return p;
    }
  }
}

// Intuitively, we are checking the parallel planes of each axis and seeing if
// the ray intersects the box that those 3 pairs of planes form. We are
// considered inside the box if the ray is within each of the 3 intervals at the
// same time, and exited the box as soon as we are outside of any one of the
// intervals. t_min tracks the latest entry point into the box. t_max tracks the
// earliest exit point from the box. After all the checks, t_min and t_max
// describe the interval where the ray is inside the box.
inline __device__ bool hit_aabb(const float3 &bmin, const float3 &bmax,
                                const Ray &r, float t_min, float t_max) {
  for (int axis = 0; axis < 3; ++axis) {
    const float origin = (axis == 0)   ? r.origin.x
                         : (axis == 1) ? r.origin.y
                                       : r.origin.z;
    const float dir = (axis == 0) ? r.dir.x : (axis == 1) ? r.dir.y : r.dir.z;
    float minv = (axis == 0) ? bmin.x : (axis == 1) ? bmin.y : bmin.z;
    float maxv = (axis == 0) ? bmax.x : (axis == 1) ? bmax.y : bmax.z;

    const float inv_dir = 1.0f / dir;
    float t0 = (minv - origin) * inv_dir;
    float t1 = (maxv - origin) * inv_dir;
    if (inv_dir < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }

    t_min = fmaxf(t_min, t0);
    t_max = fminf(t_max, t1);
    if (t_max <= t_min) {
      return false;
    }
  }
  return true;
}

inline __device__ bool hit_sphere(const GpuSphere &s, const Ray &r, float t_min,
                                  float t_max, Hit &out_hit) {
  float3 oc = sub3(s.center, r.origin);
  float a = len_sq3(r.dir);
  float h = dot3(r.dir, oc);
  float c = len_sq3(oc) - s.radius * s.radius;
  float disc = h * h - a * c;

  if (disc < 0.0f) {
    return false;
  }

  float sqrtd = sqrtf(disc);

  float root = (h - sqrtd) / a;
  if (root <= t_min || root >= t_max) {
    root = (h + sqrtd) / a;
    if (root <= t_min || root >= t_max) {
      return false;
    }
  }

  out_hit.t = root;
  out_hit.p = ray_at(r, root);
  float3 outward_normal = div3(sub3(out_hit.p, s.center), s.radius);
  set_face_normal(out_hit, r, outward_normal);
  out_hit.material_index = s.material_index;

  return true;
}

__noinline__ __device__ bool hit_scene(const GpuScene &scene, const Ray &r,
                                       float t_min, float t_max, Hit &rec) {
  if (scene.bvh_nodes == nullptr || scene.bvh_node_count == 0 ||
      scene.bvh_primitive_indices == nullptr) {
    return false;
  }

  Hit tmp;
  bool hit_anything = false;
  float closest = t_max;

  uint32_t node_index = 0;
  while (node_index < static_cast<uint32_t>(scene.bvh_node_count)) {
    const BVHNode &node = scene.bvh_nodes[node_index];
    const bool is_leaf = (node.flags & 1u) != 0u;

    if (!hit_aabb(node.aabb_min, node.aabb_max, r, t_min, closest)) {
      node_index = is_leaf ? (node_index + 1u) : node.right;
      continue;
    }

    if (is_leaf) {
      const uint32_t first = node.left;
      const uint32_t count = node.right;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t sphere_index = scene.bvh_primitive_indices[first + i];
        if (hit_sphere(scene.spheres[sphere_index], r, t_min, closest, tmp)) {
          hit_anything = true;
          closest = tmp.t;
          rec = tmp;
        }
      }
      node_index += 1u;
    } else {
      node_index = node.left;
    }
  }

  return hit_anything;
}

inline __device__ float reflectance(float cosine, float ref_idx) {
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

inline __device__ bool scatter(const GpuMaterial &mat, const Ray &r_in,
                               const Hit &hit, RngState &rng,
                               float3 &attenuation, Ray &scattered) {

  if (mat.type == MAT_LAMBERTIAN) {
    float3 scatter_dir = add3(hit.normal, random_unit_vector(rng));
    if (len_sq3(scatter_dir) < 1e-8f) {
      scatter_dir = hit.normal;
    }
    scattered.origin = hit.p;
    scattered.dir = scatter_dir;
    attenuation = mat.albedo;
    return true;
  }

  if (mat.type == MAT_METAL) {
    float3 reflected = reflect3(unit3(r_in.dir), hit.normal);
    reflected = add3(reflected, mul3(random_unit_vector(rng), mat.fuzz));
    scattered.origin = hit.p;
    scattered.dir = reflected;
    attenuation = mat.albedo;
    return dot3(scattered.dir, hit.normal) > 0.0f;
  }

  if (mat.type == MAT_DIELECTRIC) {
    attenuation = make_vec3(1.0f, 1.0f, 1.0f);
    float eta_ratio = hit.front_face ? (1.0f / mat.ref_idx) : mat.ref_idx;

    float3 unit_dir = unit3(r_in.dir);
    float cos_theta = fminf(dot3(mul3(unit_dir, -1.0f), hit.normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    bool cannot_refract = eta_ratio * sin_theta > 1.0f;
    float3 direction;
    if (cannot_refract ||
        reflectance(cos_theta, eta_ratio) > rng_next_f32(rng)) {
      direction = reflect3(unit_dir, hit.normal);
    } else {
      direction = refract3(unit_dir, hit.normal, eta_ratio);
    }

    scattered.origin = hit.p;
    scattered.dir = direction;
    return true;
  }

  return false;
}

inline __device__ float3 sky_color(const Ray &r) {
  float3 unit_dir = unit3(r.dir);
  float a = 0.5f * (unit_dir.y + 1.0f);
  float3 c0 = make_vec3(1.0f, 1.0f, 1.0f);
  float3 c1 = make_vec3(0.5f, 0.7f, 1.0f);
  return add3(mul3(c0, 1.0f - a), mul3(c1, a));
}

inline __device__ Ray get_ray(const GpuCamera &cam, int px, int py,
                              RngState &rng) {
  float ox = rng_next_f32(rng) - 0.5f;
  float oy = rng_next_f32(rng) - 0.5f;

  float3 pixel_sample =
      add3(cam.pixel00_loc,
           add3(mul3(cam.pixel_delta_u, static_cast<float>(px) + ox),
                mul3(cam.pixel_delta_v, static_cast<float>(py) + oy)));

  float3 origin = cam.center;
  if (cam.defocus_angle > 0.0f) {
    float3 p = random_in_unit_disk(rng);
    origin = add3(cam.center, add3(mul3(cam.defocus_disk_u, p.x),
                                   mul3(cam.defocus_disk_v, p.y)));
  }

  Ray r;
  r.origin = origin;
  r.dir = sub3(pixel_sample, origin);
  return r;
}

inline __device__ uchar3 to_rgb8(const float3 &c) {
  float r = sqrtf(fmaxf(c.x, 0.0f));
  float g = sqrtf(fmaxf(c.y, 0.0f));
  float b = sqrtf(fmaxf(c.z, 0.0f));

  unsigned char rb =
      static_cast<unsigned char>(256.0f * clampf(r, 0.0f, 0.999f));
  unsigned char gb =
      static_cast<unsigned char>(256.0f * clampf(g, 0.0f, 0.999f));
  unsigned char bb =
      static_cast<unsigned char>(256.0f * clampf(b, 0.0f, 0.999f));

  return make_uchar3(rb, gb, bb);
}

} // namespace

__global__ void init_rng_kernel(RngState *rng, int width, int height,
                                uint32_t seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int idx = y * width + x;
  rng_seed(rng[idx], seed, static_cast<uint32_t>(idx));
}

__global__ void init_paths_kernel(PathState *paths, GpuCamera cam,
                                  const RngState *rng_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.image_width || y >= cam.image_height) {
    return;
  }

  int idx = y * cam.image_width + x;
  PathState path{};
  path.rng = rng_states[idx];
  path.ray = get_ray(cam, x, y, path.rng);
  path.throughput = make_vec3(1.0f, 1.0f, 1.0f);
  path.radiance = make_vec3(0.0f, 0.0f, 0.0f);
  path.pixel_index = idx;
  path.depth = 0;
  path.active = 1;
  paths[idx] = path;
}

__global__ void trace_paths_kernel(PathState *paths, PathHit *hits,
                                   GpuCamera cam, GpuScene scene) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.image_width || y >= cam.image_height) {
    return;
  }

  int idx = y * cam.image_width + x;
  PathState &path = paths[idx];
  PathHit &hit = hits[idx];

  if (!path.active) {
    hit.hit = 0;
    return;
  }

  Hit rec;
  if (!hit_scene(scene, path.ray, 0.001f, 1e30f, rec)) {
    path.radiance = add3(path.radiance,
                         mul3(path.throughput, sky_color(path.ray)));
    path.active = 0;
    hit.hit = 0;
    return;
  }

  hit.rec = rec;
  hit.hit = 1;
}

__global__ void shade_paths_kernel(PathState *paths, const PathHit *hits,
                                   GpuCamera cam, GpuScene scene) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.image_width || y >= cam.image_height) {
    return;
  }

  int idx = y * cam.image_width + x;
  PathState &path = paths[idx];
  if (!path.active || !hits[idx].hit) {
    return;
  }

  const Hit &rec = hits[idx].rec;
  const GpuMaterial &mat = scene.materials[rec.material_index];

  float3 attenuation;
  Ray scattered;
  if (!scatter(mat, path.ray, rec, path.rng, attenuation, scattered)) {
    path.active = 0;
    return;
  }

  path.throughput = mul3(path.throughput, attenuation);
  path.ray = scattered;
  path.depth += 1;
  if (path.depth >= cam.max_depth) {
    path.active = 0;
  }
}

__global__ void accumulate_paths_kernel(const PathState *paths,
                                        RngState *rng_states,
                                        float3 *accum, GpuCamera cam) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.image_width || y >= cam.image_height) {
    return;
  }

  int idx = y * cam.image_width + x;
  const PathState &path = paths[idx];
  accum[path.pixel_index] = add3(accum[path.pixel_index], path.radiance);
  rng_states[path.pixel_index] = path.rng;
}

__global__ void resolve_framebuffer_kernel(uchar3 *out_rgb, const float3 *accum,
                                           GpuCamera cam) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= cam.image_width || y >= cam.image_height) {
    return;
  }

  int idx = y * cam.image_width + x;
  float inv_spp = 1.0f / static_cast<float>(cam.samples_per_pixel);
  out_rgb[idx] = to_rgb8(mul3(accum[idx], inv_spp));
}

cudaError_t launch_init_rng(RngState *d_rng, int image_width, int image_height,
                            uint32_t seed, cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((image_width + block.x - 1) / block.x,
            (image_height + block.y - 1) / block.y);

  init_rng_kernel<<<grid, block, 0, stream>>>(d_rng, image_width, image_height,
                                              seed);
  return cudaGetLastError();
}

cudaError_t launch_render(uchar3 *d_framebuffer, GpuCamera cam, GpuScene scene,
                          RngState *d_rng, cudaStream_t stream) {

  const int pixel_count = cam.image_width * cam.image_height;

  PathState *d_paths = nullptr;
  PathHit *d_hits = nullptr;
  float3 *d_accum = nullptr;

  auto cleanup = [&]() {
    if (d_accum) {
      cudaFree(d_accum);
    }
    if (d_hits) {
      cudaFree(d_hits);
    }
    if (d_paths) {
      cudaFree(d_paths);
    }
  };

  cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_paths),
                               pixel_count * sizeof(PathState));
  if (err != cudaSuccess) {
    cleanup();
    return err;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_hits),
                   pixel_count * sizeof(PathHit));
  if (err != cudaSuccess) {
    cleanup();
    return err;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_accum),
                   pixel_count * sizeof(float3));
  if (err != cudaSuccess) {
    cleanup();
    return err;
  }

  err = cudaMemsetAsync(d_accum, 0, pixel_count * sizeof(float3), stream);
  if (err != cudaSuccess) {
    cleanup();
    return err;
  }

  dim3 block(16, 16);
  dim3 grid((cam.image_width + block.x - 1) / block.x,
            (cam.image_height + block.y - 1) / block.y);

  for (int sample = 0; sample < cam.samples_per_pixel; ++sample) {
    init_paths_kernel<<<grid, block, 0, stream>>>(d_paths, cam, d_rng);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return err;
    }

    for (int bounce = 0; bounce < cam.max_depth; ++bounce) {
      trace_paths_kernel<<<grid, block, 0, stream>>>(d_paths, d_hits, cam,
                                                     scene);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        cleanup();
        return err;
      }

      shade_paths_kernel<<<grid, block, 0, stream>>>(d_paths, d_hits, cam,
                                                     scene);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        cleanup();
        return err;
      }
    }

    accumulate_paths_kernel<<<grid, block, 0, stream>>>(d_paths, d_rng,
                                                        d_accum, cam);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cleanup();
      return err;
    }
  }

  resolve_framebuffer_kernel<<<grid, block, 0, stream>>>(d_framebuffer,
                                                          d_accum, cam);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    cleanup();
    return err;
  }

  err = cudaStreamSynchronize(stream);
  cleanup();
  return err;
}
