#include "renderer.cuh"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::cout;
using std::floor;

namespace {

float deg_to_rad(float deg) { return deg * 3.14159265358979323846f / 180.0f; }

float randf(std::mt19937 &rng) {
  static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng);
}

float randf(std::mt19937 &rng, float lo, float hi) {
  return lo + (hi - lo) * randf(rng);
}

float3 random_color(std::mt19937 &rng) {
  return make_vec3(randf(rng), randf(rng), randf(rng));
}

float3 random_color(std::mt19937 &rng, float lo, float hi) {
  return make_vec3(randf(rng, lo, hi), randf(rng, lo, hi), randf(rng, lo, hi));
}

std::string get_output_filename(const std::string &scene_name) {
  return std::string("images/") + scene_name + "_cuda.png";
}

GpuMaterial make_lambertian(const float3 &albedo) {
  GpuMaterial m{};
  m.type = MAT_LAMBERTIAN;
  m.albedo = albedo;
  m.fuzz = 0.0f;
  m.ref_idx = 1.0f;
  return m;
}

// GpuMaterial make_metal(const float3 &albedo, float fuzz)
// {
//     GpuMaterial m{};
//     m.type = MAT_METAL;
//     m.albedo = albedo;
//     m.fuzz = fuzz;
//     m.ref_idx = 1.0f;
//     return m;
// }

// GpuMaterial make_dielectric(float ref_idx)
// {
//     GpuMaterial m{};
//     m.type = MAT_DIELECTRIC;
//     m.albedo = make_vec3(1.0f, 1.0f, 1.0f);
//     m.fuzz = 0.0f;
//     m.ref_idx = ref_idx;
//     return m;
// }

void add_sphere(std::vector<GpuSphere> &spheres, const float3 &center,
                float radius, int mat_index) {
  GpuSphere s{};
  s.center = center;
  s.radius = radius;
  s.material_index = mat_index;
  spheres.push_back(s);
}

GpuCamera build_camera(int image_width, float aspect_ratio, int samples,
                       int max_depth, float vfov, const float3 &look_from,
                       const float3 &look_at, const float3 &vup,
                       float defocus_angle, float focus_dist) {
  GpuCamera cam{};

  int image_height = static_cast<int>(image_width / aspect_ratio);
  if (image_height < 1) {
    image_height = 1;
  }

  float theta = deg_to_rad(vfov);
  float h = std::tan(theta * 0.5f);
  float viewport_height = 2.0f * h * focus_dist;
  float viewport_width =
      viewport_height * (static_cast<float>(image_width) / image_height);

  float3 w = unit3(sub3(look_from, look_at));
  float3 u = unit3(cross3(vup, w));
  float3 v = cross3(w, u);

  float3 viewport_u = mul3(u, viewport_width);
  float3 viewport_v = mul3(v, -viewport_height);

  float3 pixel_delta_u = div3(viewport_u, static_cast<float>(image_width));
  float3 pixel_delta_v = div3(viewport_v, static_cast<float>(image_height));

  float3 viewport_upper_left =
      sub3(sub3(sub3(look_from, mul3(w, focus_dist)), div3(viewport_u, 2.0f)),
           div3(viewport_v, 2.0f));
  float3 pixel00_loc =
      add3(viewport_upper_left, mul3(add3(pixel_delta_u, pixel_delta_v), 0.5f));

  float defocus_radius =
      focus_dist * std::tan(deg_to_rad(defocus_angle * 0.5f));

  cam.image_width = image_width;
  cam.image_height = image_height;
  cam.samples_per_pixel = samples;
  cam.max_depth = max_depth;

  cam.center = look_from;
  cam.pixel00_loc = pixel00_loc;
  cam.pixel_delta_u = pixel_delta_u;
  cam.pixel_delta_v = pixel_delta_v;

  cam.defocus_disk_u = mul3(u, defocus_radius);
  cam.defocus_disk_v = mul3(v, defocus_radius);
  cam.defocus_angle = defocus_angle;

  return cam;
}

struct SceneData {
  GpuCamera camera;
  std::vector<GpuSphere> spheres;
  std::vector<GpuMaterial> materials;
};

SceneData setup_simple_scene(int samples) {
  SceneData s{};
  s.materials.reserve(5);

  int ground = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_metal(make_vec3(0.5f, 0.5f, 0.5f), 0.1f));

  int center = static_cast<int>(s.materials.size());
  s.materials.push_back(make_lambertian(make_vec3(0.1f, 0.2f, 0.5f)));

  int left = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_dielectric(1.5f));

  int bubble = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_dielectric(1.0f / 1.5f));

  int right = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_metal(make_vec3(0.8f, 0.6f, 0.2f), 1.0f));

  add_sphere(s.spheres, make_vec3(0.0f, -100.5f, -1.0f), 100.0f, ground);
  add_sphere(s.spheres, make_vec3(0.0f, 0.0f, -1.2f), 0.5f, center);
  add_sphere(s.spheres, make_vec3(-1.0f, 0.0f, -1.0f), 0.5f, left);
  add_sphere(s.spheres, make_vec3(-1.0f, 0.0f, -1.0f), 0.4f, bubble);
  add_sphere(s.spheres, make_vec3(1.0f, 0.0f, -1.0f), 0.5f, right);

  int image_width = 400;
  float aspect_ratio = 16.0f / 9.0f;
  int max_depth = 50;
  float vfov = 40.0f;
  float3 look_from = make_vec3(0.0f, 0.0f, 1.0f);
  float3 look_at = make_vec3(0.0f, 0.0f, -1.0f);
  float3 vup = make_vec3(0.0f, 1.0f, 0.0f);
  float defocus_angle = 10.0f;
  float focus_dist = len3(sub3(look_from, look_at));

  s.camera = build_camera(image_width, aspect_ratio, samples, max_depth, vfov,
                          look_from, look_at, vup, defocus_angle, focus_dist);

  return s;
}

SceneData setup_cover_scene(int samples) {
  SceneData s{};
  std::mt19937 rng(1337u);
  float aspect_ratio = 16.0f / 9.0f;

  // Determine the next material index before adding the material to the list.
  int ground_index = static_cast<int>(s.materials.size());
  s.materials.push_back(make_lambertian(make_vec3(0.5f, 0.5f, 0.5f)));
  add_sphere(s.spheres, make_vec3(0.0f, -5000.0f, 0.0f), 5000.0f, ground_index);

  int grid_size = 22;

  // Make 22x22 = 484 small spheres
  for (int a = -floor(grid_size / 2.0f); a < floor(grid_size / 2.0f); ++a) {
    for (int b = -floor(grid_size / 2.0f); b < floor(grid_size / 2.0f); ++b) {

      // float choose_mat = randf(rng);
      float3 center = make_vec3(static_cast<float>(a) + 0.9f * randf(rng), 0.2,
                                static_cast<float>(b) + 0.9f * randf(rng));

      float3 avoid = make_vec3(4.0f, 0.2f, 0.0f);
      if (len3(sub3(center, avoid)) <= 0.9f) {
        b--;
        continue;
      }

      if (true) //(choose_mat < 1.0f)
      {
        int mat = static_cast<int>(s.materials.size());
        s.materials.push_back(make_lambertian(
            mul3(random_color(rng, 0.5f, 1.0f), random_color(rng))));
        add_sphere(s.spheres, center, 0.2f, mat);
      }
      // else if (choose_mat < 0.95f)
      // {
      //     int mat = static_cast<int>(s.materials.size());
      //     s.materials.push_back(make_metal(random_color(rng, 0.5f, 1.0f),
      //     randf(rng, 0.0f, 0.5f))); add_sphere(s.spheres, center, 0.2f,
      //     mat);
      // }
      // else
      // {
      //     int mat = static_cast<int>(s.materials.size());
      //     s.materials.push_back(make_dielectric(1.5f));
      //     add_sphere(s.spheres, center, 0.2f, mat);
      // }
    }
  }

  // Make 3 large spheres
  for (int i = 0; i < 3; ++i) {
    int new_mat = static_cast<int>(s.materials.size());
    s.materials.push_back(make_lambertian(
        mul3(random_color(rng, 0.5f, 1.0f), random_color(rng))));
    add_sphere(s.spheres, make_vec3(-4.0f + 4 * i, 1.0f, 0.0f), 1.0f, new_mat);
  }

  cout << "Number of scene objects: " << s.spheres.size() << "\n";
  // int m1 = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_dielectric(1.5f));
  // add_sphere(s.spheres, make_vec3(0.0f, 1.0f, 0.0f), 1.0f, m1);

  // int m2 = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_lambertian(mul3(random_color(rng),
  // random_color(rng)))); add_sphere(s.spheres, make_vec3(-4.0f, 1.0f,
  // 0.0f), 1.0f, m2);

  // int m3 = static_cast<int>(s.materials.size());
  // s.materials.push_back(make_metal(make_vec3(0.7f, 0.6f, 0.5f), 0.0f));
  // add_sphere(s.spheres, make_vec3(4.0f, 1.0f, 0.0f), 1.0f, m3);

  // s.camera = build_camera(1200, aspect_ratio, samples, 10, 20.0f,
  //                         make_vec3(13.0f, 2.0f, 3.0f), // from
  //                         make_vec3(0.0f, 0.0f, 0.0f), // at
  //                         make_vec3(0.0f, 1.0f, 0.0f),   // up
  //                         0.6f, 10.0f);

  s.camera = build_camera(1200, aspect_ratio, samples, 10, 20.0f,
                          make_vec3(13.0f, 2.0f, 3.0f), // from
                          make_vec3(0.0f, 0.0f, 0.0f),  // at
                          make_vec3(0.0f, 1.0f, 0.0f),  // up
                          0.0f, 10.0f);

  return s;
}

SceneData setup_spiral_shere(int samples) {
  SceneData s{};
  std::mt19937 rng(1337u);
  float aspect_ratio = 16.0f / 9.0f;

  // Add ground
  int ground_index = static_cast<int>(s.materials.size());
  s.materials.push_back(make_lambertian(make_vec3(0.5f, 0.5f, 0.5f)));
  add_sphere(s.spheres, make_vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_index);

  // Create the geometry of the spiral sphere
  float radius_big = 15.0f; // Height of the arrangement is twice this value.
  float radius_small = 0.3f;
  int num_spheres = 2000;
  int turns = 20;
  float y_offset = radius_big + radius_small;

  auto speed = [&](float t) {
    float phi = acos(1.0f - 2.0f * t);
    float dphi_dt = 2.0f / sqrt(1.0f - pow(1.0f - 2.0f * t, 2.0f));
    float dtheta_dt = 2.0f * PI * turns;
    return sqrt(pow(dphi_dt, 2.0f) + pow(sin(phi) * dtheta_dt, 2.0f));
  };

  auto make_equal_arclength_params = [&](int num_spheres, int turns) {
    int steps = 10000; // also increase this back to 10000
    std::vector<float> cumulative(steps + 1, 0.0f);

    // never reach the poles entirely to avoid math complications. Barely
    // noticable visually if at all.
    float t_min = 0.01f;
    float t_max = 0.99f;

    for (int i = 1; i <= steps; ++i) {
      float t = t_min + (t_max - t_min) * ((float)i / steps);
      float dt = (t_max - t_min) / steps;
      cumulative[i] = cumulative[i - 1] + speed(t) * dt;
    }
    float total = cumulative[steps];

    std::vector<float> params;
    params.reserve(num_spheres);

    int j = 0;
    for (int i = 0; i < num_spheres; ++i) {
      float target = total * ((float)i / (float)(num_spheres - 1));
      while (j < steps && cumulative[j + 1] < target) {
        ++j;
      }
      float t_raw = float(j) / steps;
      params.push_back(t_min +
                       (t_max - t_min) * t_raw); // remap back to [t_min, t_max]
    }

    return params;
  };

  auto params = make_equal_arclength_params(num_spheres, turns);

  std::vector<float3> rainbow = {
      make_vec3(0.918f, 0.047f, 0.047f), // Red
      make_vec3(1.000f, 0.498f, 0.000f), // Orange
      make_vec3(1.000f, 0.929f, 0.000f), // Yellow
      make_vec3(0.133f, 0.694f, 0.298f), // Green
      make_vec3(0.063f, 0.329f, 0.780f), // Blue
      make_vec3(0.294f, 0.000f, 0.510f), // Indigo
      make_vec3(0.580f, 0.000f, 0.827f), // Violet
  };

  // Loop to build the spiral shape
  for (int i = 0; i < num_spheres; ++i) {
    float t = params[i];
    float phi = acos(1.0f - 2.0f * t);
    float theta = 2.0f * PI * turns * t;

    float x = radius_big * sin(phi) * cos(theta);
    float y = radius_big * cos(phi);
    float z = radius_big * sin(phi) * sin(theta);

    // define location of the component sphere
    float3 center = {x, y, z};
    // apply rotation to the large object
    center = rotate_vec3(center, make_vec3(0.0f, 0.0f, 1.0f), 30.0f);
    // apply y offset so that the object is not centered around y = 0
    center.y += y_offset;

    // Define new material index
    int material = static_cast<int>(s.materials.size());
    // Define new semi-random color
    // s.materials.push_back(make_lambertian(
    //     mul3(random_color(rng, 0.5f, 1.0f), random_color(rng))));
    s.materials.push_back(make_lambertian(
        make_vec3(rainbow[i % 7].x, rainbow[i % 7].y, rainbow[i % 7].z)));
    // Add sphere to the list
    add_sphere(s.spheres, center, radius_small, material);
  }

  cout << "Number of scene objects: " << s.spheres.size() << "\n";

  // Set camera parameters
  s.camera = build_camera(1200, aspect_ratio, samples, 10, 45.0f,
                          make_vec3(-21.5f, 45.0f, 21.5f), // from
                          make_vec3(0.0f, 15.0f, 0.0f),    // at
                          make_vec3(0.0f, 1.0f, 0.0f),     // up
                          0.0f, 10.0f);

  return s;
}

} // namespace

int main(int argc, char **argv) {
  std::string scene_name = "cover";
  int samples = 10;

  if (argc >= 2) {
    scene_name = argv[1];
  }
  if (argc >= 3) {
    try {
      samples = std::stoi(argv[2]);
      if (samples < 1) {
        std::cerr << "samples must be >= 1\n";
        return 1;
      }
    } catch (const std::exception &) {
      std::cerr << "Invalid samples value: " << argv[2]
                << " (must be integer)\n";
      return 1;
    }
  }

  SceneData scene;
  if (scene_name == "cover") {
    scene = setup_cover_scene(samples);
  } else if (scene_name == "simple") {
    scene = setup_simple_scene(samples);
  } else if (scene_name == "spiral") {
    scene = setup_spiral_shere(samples);
  } else {
    std::cerr << "Invalid scene: " << scene_name
              << " (use cover|simple|spiral)\n";
    return 1;
  }

  // Print settings used for this particular run
  cout << "Scene: " << scene_name << "\n";
  cout << "Samples per pixel : " << samples << "\n";

  double cuda_seconds = 0.0;
  std::string cuda_error;
  std::string output = get_output_filename(scene_name);

  if (!render_cuda_scene(scene.camera, scene.spheres, scene.materials, output,
                         cuda_seconds, cuda_error)) {
    std::cerr << "CUDA render failed: " << cuda_error << "\n";
    return 1;
  }

  cout << "CUDA render time: " << cuda_seconds << " s\n";
  return 0;
}
