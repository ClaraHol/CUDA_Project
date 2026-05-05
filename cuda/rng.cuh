#pragma once

#include <stdint.h>

struct RngState
{
    uint32_t state;
};

inline __host__ __device__ uint32_t wang_hash(uint32_t x)
{
    x = (x ^ 61u) ^ (x >> 16);
    x *= 9u;
    x = x ^ (x >> 4);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15);
    return x;
}

inline __host__ __device__ void rng_seed(RngState &rng, uint32_t seed, uint32_t sequence)
{
    rng.state = wang_hash(seed ^ (sequence * 747796405u + 2891336453u));
    if (rng.state == 0u)
    {
        rng.state = 1u;
    }
}

inline __host__ __device__ uint32_t rng_next_u32(RngState &rng)
{
    uint32_t x = rng.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng.state = x;
    return x;
}

inline __host__ __device__ float rng_next_f32(RngState &rng)
{
    return (rng_next_u32(rng) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}
