#pragma once

#include "cuda_types.cuh"

struct GpuCamera
{
    int image_width;
    int image_height;
    int samples_per_pixel;
    int max_depth;

    float3 center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;

    float3 defocus_disk_u;
    float3 defocus_disk_v;
    float defocus_angle;
};
