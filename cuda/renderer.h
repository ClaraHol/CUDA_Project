#pragma once

#include <vector>
#include <string>

#include "camera.cuh"
#include "types.cuh"

bool render_cuda_scene(
    const GpuCamera &cam,
    const std::vector<GpuSphere> &spheres,
    const std::vector<GpuMaterial> &materials,
    const std::string &output_path,
    double &elapsed_seconds,
    std::string &error_message);
