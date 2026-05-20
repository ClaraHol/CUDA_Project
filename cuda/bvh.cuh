#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

struct BVHNode
{
    float3 aabb_min;
    float3 aabb_max;

    // Internal node:
    // left = first child node index (preorder next node)
    // right = escape index (node after this subtree)
    //
    // Leaf node:
    // left = first primitive index in primitive_indices
    // right = primitive count in this leaf
    uint32_t left;
    uint32_t right;

    // bit 0: is_leaf
    // bits 1-2: split axis
    uint32_t flags;
    uint32_t pad;
};

struct BVHBuildResult
{
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> primitive_indices;
};
