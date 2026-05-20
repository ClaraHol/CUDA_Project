#pragma once
#include <vector>
#include "bvh.cuh"
#include "types.cuh"

BVHBuildResult build_bvh_for_spheres(const std::vector<GpuSphere> &spheres, int leaf_size = 4);