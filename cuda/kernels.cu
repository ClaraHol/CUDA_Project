#include "bvh.cuh"
#include "kernels.cuh"

#include <math.h>

namespace {

// ---------------------------------------------------------------------------
// RNG helpers
// ---------------------------------------------------------------------------

// Direct spherical sampling — no rejection loop, unit length by construction.
// Samples uniformly on the unit sphere via the analytic parameterisation:
//   cos_theta uniform in [-1,1], phi uniform in [0, 2pi].
inline __device__ float3 random_unit_vector(RngState &rng) {
  // cos_theta in [-1, 1]
  float cos_theta = 2.0f * rng_next_f32(rng) - 1.0f;
  float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = 2.0f * PI * rng_next_f32(rng);
  float sin_phi, cos_phi;
  sincosf(phi, &sin_phi, &cos_phi);
  return make_vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

// Concentric disk sampling — single pass, no rejection.
// Maps the unit square to the unit disk while preserving area.
inline __device__ float2 random_in_unit_disk(RngState &rng) {
  float r = sqrtf(rng_next_f32(rng));
  float theta = 2.0f * PI * rng_next_f32(rng);
  float sin_t, cos_t;
  sincosf(theta, &sin_t, &cos_t);
  return make_float2(r * cos_t, r * sin_t);
}

// ---------------------------------------------------------------------------
// AABB intersection — three axes manually unrolled.
// inv_dir is pre-negated by swapping t0/t1 when inv_dir < 0, which avoids
// a branch per axis by using fminf/fmaxf on the already-ordered values.
// ---------------------------------------------------------------------------
inline __device__ bool hit_aabb(const float4 &bmin, const float4 &bmax,
                                const Ray &r, float t_min, float t_max) {
  // X axis
  {
    const float inv_dir = 1.0f / r.dir.x;
    float t0 = (bmin.x - r.origin.x) * inv_dir;
    float t1 = (bmax.x - r.origin.x) * inv_dir;
    if (inv_dir < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    t_min = fmaxf(t_min, t0);
    t_max = fminf(t_max, t1);
    if (t_max <= t_min)
      return false;
  }
  // Y axis
  {
    const float inv_dir = 1.0f / r.dir.y;
    float t0 = (bmin.y - r.origin.y) * inv_dir;
    float t1 = (bmax.y - r.origin.y) * inv_dir;
    if (inv_dir < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    t_min = fmaxf(t_min, t0);
    t_max = fminf(t_max, t1);
    if (t_max <= t_min)
      return false;
  }
  // Z axis
  {
    const float inv_dir = 1.0f / r.dir.z;
    float t0 = (bmin.z - r.origin.z) * inv_dir;
    float t1 = (bmax.z - r.origin.z) * inv_dir;
    if (inv_dir < 0.0f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    t_min = fmaxf(t_min, t0);
    t_max = fminf(t_max, t1);
    if (t_max <= t_min)
      return false;
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

  if (disc < 0.0f)
    return false;

  float sqrtd = sqrtf(disc);
  float root = (h - sqrtd) / a;
  if (root <= t_min || root >= t_max) {
    root = (h + sqrtd) / a;
    if (root <= t_min || root >= t_max)
      return false;
  }

  out_hit.t = root;
  out_hit.p = ray_at(r, root);
  float3 outward_normal = div3(sub3(out_hit.p, s.center), s.radius);
  set_face_normal(out_hit, r, outward_normal);
  out_hit.material_index = s.material_index;
  return true;
}

// ---------------------------------------------------------------------------
// BVH traversal — __ldg() for read-only node loads.
// The node struct fields are fetched into local variables once per iteration
// to avoid repeated global memory loads from the same address.
// ---------------------------------------------------------------------------
inline __device__ bool hit_scene(const GpuScene &scene, const Ray &r,
                                 float t_min, float t_max, Hit &rec) {
  if (scene.bvh_nodes == nullptr || scene.bvh_node_count == 0 ||
      scene.bvh_primitive_indices == nullptr) {
    return false;
  }

  Hit tmp;
  bool hit_anything = false;
  float closest = t_max;

  uint32_t node_index = 0;
  const uint32_t node_count = static_cast<uint32_t>(scene.bvh_node_count);

  while (node_index < node_count) {
    // Prefetch node fields via texture cache — BVH is read-only during
    // traversal.
    const BVHNode *np = scene.bvh_nodes + node_index;
    const float4 aabb_min = __ldg(&np->aabb_min);
    const float4 aabb_max = __ldg(&np->aabb_max);
    const uint32_t left = __ldg(&np->left);
    const uint32_t right = __ldg(&np->right);
    const uint32_t flags = __ldg(&np->flags);
    const bool is_leaf = (flags & 1u) != 0u;

    if (!hit_aabb(aabb_min, aabb_max, r, t_min, closest)) {
      node_index = is_leaf ? (node_index + 1u) : right;
      continue;
    }

    if (is_leaf) {
      const uint32_t first = left;
      const uint32_t count = right;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t si = __ldg(&scene.bvh_primitive_indices[first + i]);
        if (hit_sphere(scene.spheres[si], r, t_min, closest, tmp)) {
          hit_anything = true;
          closest = tmp.t;
          rec = tmp;
        }
      }
      node_index += 1u;
    } else {
      node_index = left;
    }
  }

  return hit_anything;
}

// ---------------------------------------------------------------------------
// Scatter — Lambertian only.
// Only the fields scatter actually needs are extracted from Hit:
//   rec.normal         — scatter direction base
//   rec.p              — new ray origin
//   rec.material_index — already consumed before scatter is called
// rec.t and rec.front_face are not read here.
// ---------------------------------------------------------------------------
inline __device__ bool scatter(const GpuMaterial &mat, const float3 &hit_p,
                               const float3 &hit_normal, RngState &rng,
                               float3 &attenuation, Ray &scattered) {
  float3 scatter_dir = add3(hit_normal, random_unit_vector(rng));
  // Guard against degenerate scatter direction when random vector cancels
  // normal.
  if (len_sq3(scatter_dir) < 1e-8f) {
    scatter_dir = hit_normal;
  }
  scattered.origin = hit_p;
  scattered.dir = scatter_dir;
  attenuation = mat.albedo;
  return true;
}

inline __device__ float3 sky_color(const Ray &r) {
  float3 unit_dir = unit3(r.dir);
  float a = 0.5f * (unit_dir.y + 1.0f);
  float3 c0 = make_vec3(1.0f, 1.0f, 1.0f);
  float3 c1 = make_vec3(0.5f, 0.7f, 1.0f);
  return add3(mul3(c0, 1.0f - a), mul3(c1, a));
}

// ---------------------------------------------------------------------------
// Primary ray generation.
// Defocus blur samples use the rejection-free disk sampler.
// ---------------------------------------------------------------------------
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
    float2 p = random_in_unit_disk(rng);
    origin = add3(cam.center, add3(mul3(cam.defocus_disk_u, p.x),
                                   mul3(cam.defocus_disk_v, p.y)));
  }

  Ray r;
  r.origin = origin;
  r.dir = sub3(pixel_sample, origin);
  return r;
}

inline __device__ uchar3 to_rgb8(const float3 &c) {
  // Gamma-2 encode (sqrt) and convert to 8-bit.
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

// ---------------------------------------------------------------------------
// Megakernel (retained for reference / comparison)
// ---------------------------------------------------------------------------
inline __device__ float3 ray_color_iterative(const Ray &initial_ray,
                                             int max_depth,
                                             const GpuScene &scene,
                                             RngState &rng) {
  Ray ray = initial_ray;
  float3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
  float3 radiance = make_vec3(0.0f, 0.0f, 0.0f);

  for (int bounce = 0; bounce < max_depth; ++bounce) {
    Hit rec;
    if (!hit_scene(scene, ray, 0.001f, 1e30f, rec)) {
      radiance = add3(radiance, mul3(throughput, sky_color(ray)));
      break;
    }
    const GpuMaterial &mat = scene.materials[rec.material_index];
    float3 attenuation;
    Ray scattered;
    if (!scatter(mat, rec.p, rec.normal, rng, attenuation, scattered))
      break;
    throughput = mul3(throughput, attenuation);
    ray = scattered;
  }
  return radiance;
}

} // namespace

// ---------------------------------------------------------------------------
// Megakernel launch (retained)
// ---------------------------------------------------------------------------

__global__ void init_rng_kernel(RngState *rng, int width, int height,
                                uint32_t seed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;
  int idx = y * width + x;
  rng_seed(rng[idx], seed, static_cast<uint32_t>(idx));
}

__global__ void render_kernel(uchar3 *out_rgb, GpuCamera cam, GpuScene scene,
                              RngState *rng_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam.image_width || y >= cam.image_height)
    return;

  int idx = y * cam.image_width + x;
  RngState rng = rng_states[idx];
  float3 accum = make_vec3(0.0f, 0.0f, 0.0f);

  for (int s = 0; s < cam.samples_per_pixel; ++s) {
    Ray ray = get_ray(cam, x, y, rng);
    accum = add3(accum, ray_color_iterative(ray, cam.max_depth, scene, rng));
  }

  float inv_spp = 1.0f / static_cast<float>(cam.samples_per_pixel);
  out_rgb[idx] = to_rgb8(mul3(accum, inv_spp));
  rng_states[idx] = rng;
}

cudaError_t launch_init_rng(RngState *d_rng, int image_width, int image_height,
                            uint32_t seed, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((image_width + block.x - 1) / block.x,
            (image_height + block.y - 1) / block.y);
  init_rng_kernel<<<grid, block, 0, stream>>>(d_rng, image_width, image_height,
                                              seed);
  return cudaGetLastError();
}

cudaError_t launch_render(uchar3 *d_framebuffer, GpuCamera cam, GpuScene scene,
                          RngState *d_rng, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((cam.image_width + block.x - 1) / block.x,
            (cam.image_height + block.y - 1) / block.y);
  render_kernel<<<grid, block, 0, stream>>>(d_framebuffer, cam, scene, d_rng);
  return cudaGetLastError();
}

__global__ void wavefront_init_kernel(WavefrontBuffers b, GpuCamera cam,
                                      uint32_t base_seed) {
  const int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= b.total_paths)
    return;

  const int pixel = path % b.total_pixels;
  const int px = pixel % cam.image_width;
  const int py = pixel / cam.image_width;

  // Seed by flat path index - every path gets a unique sequence
  RngState rng;
  rng_seed(rng, base_seed, static_cast<uint32_t>(path));

  b.rays[path] = get_ray(cam, px, py, rng);
  b.throughput[path] = make_vec3(1.0f, 1.0f, 1.0f);
  b.active[path] = true;
  b.rng[path] = rng;
  b.pixel_index[path] = pixel;
  // pixel_index not needed — in one-sample-per-kernel layout, path == pixel.
}

// --- Bounce -----------------------------------------------------------------
// One bounce per launch. Host calls this max_depth times per sample.
// (256,4) launch bounds is default.
__global__ __launch_bounds__(256,
                             4) void wavefront_bounce_kernel(WavefrontBuffers b,
                                                             GpuScene scene) {
  const int path = blockIdx.x * blockDim.x + threadIdx.x;
  if (path >= b.total_paths)
    return;
  if (!b.active[path])
    return;

  RngState rng = b.rng[path];
  Ray ray = b.rays[path];
  const int pixel = b.pixel_index[path];

  Hit rec;
  if (!hit_scene(scene, ray, 0.001f, 1e30f, rec)) {
    // Missed — accumulate sky and terminate.
    const float3 contrib = mul3(b.throughput[path], sky_color(ray));
    atomicAdd(&b.accum[pixel].x, contrib.x);
    atomicAdd(&b.accum[pixel].y, contrib.y);
    atomicAdd(&b.accum[pixel].z, contrib.z);
    b.active[path] = false;
    b.rng[path] = rng;
    return;
  }

  const GpuMaterial &mat = scene.materials[rec.material_index];
  float3 attenuation;
  Ray scattered;

  // Pass only the two fields scatter actually reads (p and normal).
  if (!scatter(mat, rec.p, rec.normal, rng, attenuation, scattered)) {
    b.active[path] = false;
    b.rng[path] = rng;
    return;
  }

  b.throughput[path] = mul3(b.throughput[path], attenuation);
  b.rays[path] = scattered;
  b.rng[path] = rng;
}

// --- Finalize ---------------------------------------------------------------
// After all samples have accumulated into b.accum, convert to gamma-corrected
// 8-bit RGB. One thread per pixel.

__global__ void wavefront_finalize_kernel(const float3 *accum, uchar3 *out,
                                          int total_pixels, float inv_spp) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= total_pixels)
    return;
  out[pixel] = to_rgb8(mul3(accum[pixel], inv_spp));
}

// --- Launch wrappers --------------------------------------------------------

// Call once before the sample loop. Zeroes the accumulator.
cudaError_t launch_wavefront_accum_clear(WavefrontBuffers &b,
                                         cudaStream_t stream) {
  return cudaMemsetAsync(b.accum, 0, b.total_pixels * sizeof(float3), stream);
}

cudaError_t launch_wavefront_init(WavefrontBuffers &b, GpuCamera cam,
                                  uint32_t seed, cudaStream_t stream) {
  const int block = 256;
  const int grid = (b.total_paths + block - 1) / block;
  wavefront_init_kernel<<<grid, block, 0, stream>>>(b, cam, seed);
  return cudaGetLastError();
}

cudaError_t launch_wavefront_bounce(WavefrontBuffers &b, GpuScene scene,
                                    cudaStream_t stream) {
  const int block = 256;
  const int grid = (b.total_paths + block - 1) / block;
  wavefront_bounce_kernel<<<grid, block, 0, stream>>>(b, scene);
  return cudaGetLastError();
}

cudaError_t launch_wavefront_finalize(const WavefrontBuffers &b, uchar3 *d_out,
                                      int samples_per_pixel,
                                      cudaStream_t stream) {
  const float inv_spp = 1.0f / static_cast<float>(samples_per_pixel);
  const int block = 256;
  const int grid = (b.total_pixels + block - 1) / block;
  wavefront_finalize_kernel<<<grid, block, 0, stream>>>(
      b.accum, d_out, b.total_pixels, inv_spp);
  return cudaGetLastError();
}
