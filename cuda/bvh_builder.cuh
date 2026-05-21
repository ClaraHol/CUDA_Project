#pragma once
#include "bvh.cuh"
#include "types.cuh"
#include <vector>

BVHBuildResult build_bvh_for_spheres(const std::vector<GpuSphere> &spheres,
                                     int leaf_size = 4);