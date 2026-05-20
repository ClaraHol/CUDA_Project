#include "bvh_builder.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace // anonymous namespace - contents only visible in this file.
{

    struct AABB
    {
        float3 minv;
        float3 maxv;
    };

    inline AABB empty_aabb()
    {
        const float inf = std::numeric_limits<float>::infinity();
        return {make_float3(inf, inf, inf), make_float3(-inf, -inf, -inf)};
    }

    inline float3 fmin3(const float3 &a, const float3 &b)
    {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }

    inline float3 fmax3(const float3 &a, const float3 &b)
    {
        return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
    }

    inline AABB merge_aabb(const AABB &a, const AABB &b)
    {
        return {fmin3(a.minv, b.minv), fmax3(a.maxv, b.maxv)};
    }

    inline AABB sphere_aabb(const GpuSphere &s)
    {
        const float3 r = make_float3(s.radius, s.radius, s.radius);
        return {
            make_float3(s.center.x - r.x, s.center.y - r.y, s.center.z - r.z),
            make_float3(s.center.x + r.x, s.center.y + r.y, s.center.z + r.z)};
    }

    class BVHBuilder
    {
    public:
        BVHBuilder(const std::vector<GpuSphere> &spheres, int leaf_size)
            : spheres_(spheres), leaf_size_(std::max(1, leaf_size))
        {
            indices_.resize(spheres_.size());
            std::iota(indices_.begin(), indices_.end(), 0u);
        }

        BVHBuildResult build()
        {
            BVHBuildResult out{}; // initialize empty
            if (spheres_.empty()) // guard against empty input
            {
                return out;
            }

            build_node(0, static_cast<int>(indices_.size()), out);
            return out;
        }

    private:
        int build_node(int begin, int end, BVHBuildResult &out)
        {
            const int node_index = static_cast<int>(out.nodes.size());
            out.nodes.push_back(BVHNode{});

            AABB bounds = empty_aabb();
            AABB centroid_bounds = empty_aabb();

            for (int i = begin; i < end; ++i)
            {
                const GpuSphere &s = spheres_[indices_[i]];
                const AABB sb = sphere_aabb(s);
                bounds = merge_aabb(bounds, sb);

                AABB cb{};
                cb.minv = s.center;
                cb.maxv = s.center;
                centroid_bounds = merge_aabb(centroid_bounds, cb);
            }

            const int count = end - begin;
            if (count <= leaf_size_)
            {
                const uint32_t first = static_cast<uint32_t>(out.primitive_indices.size());
                for (int i = begin; i < end; ++i)
                {
                    out.primitive_indices.push_back(indices_[i]);
                }

                BVHNode leaf{};
                leaf.aabb_min = bounds.minv;
                leaf.aabb_max = bounds.maxv;
                leaf.left = first;
                leaf.right = static_cast<uint32_t>(count);
                leaf.flags = 1u;
                leaf.pad = 0u;
                out.nodes[node_index] = leaf;
                return node_index;
            }

            const float3 ext = make_float3(
                centroid_bounds.maxv.x - centroid_bounds.minv.x,
                centroid_bounds.maxv.y - centroid_bounds.minv.y,
                centroid_bounds.maxv.z - centroid_bounds.minv.z);

            uint32_t axis = 0u;
            if (ext.y > ext.x && ext.y >= ext.z)
            {
                axis = 1u;
            }
            else if (ext.z > ext.x && ext.z >= ext.y)
            {
                axis = 2u;
            }

            auto centroid_on_axis = [&](uint32_t idx) -> float
            {
                const float3 c = spheres_[idx].center;
                if (axis == 0u)
                    return c.x;
                if (axis == 1u)
                    return c.y;
                return c.z;
            };

            const int mid = begin + count / 2;
            std::nth_element(
                indices_.begin() + begin,
                indices_.begin() + mid,
                indices_.begin() + end,
                [&](uint32_t a, uint32_t b)
                {
                    return centroid_on_axis(a) < centroid_on_axis(b);
                });

            const int left_child = build_node(begin, mid, out);
            build_node(mid, end, out);

            const uint32_t escape = static_cast<uint32_t>(out.nodes.size());

            BVHNode inner{};
            inner.aabb_min = bounds.minv;
            inner.aabb_max = bounds.maxv;
            inner.left = static_cast<uint32_t>(left_child);
            inner.right = escape;
            inner.flags = (axis << 1);
            inner.pad = 0u;
            out.nodes[node_index] = inner;

            return node_index;
        }

        const std::vector<GpuSphere> &spheres_;
        int leaf_size_;
        std::vector<uint32_t> indices_;
    };

} // namespace

BVHBuildResult build_bvh_for_spheres(const std::vector<GpuSphere> &spheres, int leaf_size)
{
    BVHBuilder builder(spheres, leaf_size);
    return builder.build();
}
