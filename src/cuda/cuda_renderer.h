#pragma once

#include <string>

class camera;
class hittable_list;

bool render_cuda_mvp(
    const camera& cam,
    const hittable_list& world,
    const std::string& output_path,
    double& elapsed_seconds,
    std::string& error_message);
