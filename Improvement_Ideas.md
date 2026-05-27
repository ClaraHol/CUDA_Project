Here are the things worth thinking about, roughly ordered by expected impact:

## Register Pressure

**Split the samples loop out of the kernel.** Your outer `samples_per_pixel` loop runs entirely inside one kernel launch, forcing `accum` to stay live across all samples on top of everything else. Moving this loop to the host and launching the kernel once per sample, accumulating in a float3 framebuffer in global memory, immediately removes 3 registers and simplifies the kernel considerably. This is also a prerequisite for wavefront restructuring.

**The `Hit` struct is fully live during scatter.** At the point where `scatter` is called, both the full `rec` and the new `scattered` ray and `attenuation` exist simultaneously. Consider whether all fields of `Hit` are actually needed by `scatter` or if you can narrow what gets passed through.

## BVH Traversal

**Your AABB test uses a branch inside a loop.** The axis loop in `hit_aabb` with conditional extraction of x/y/z components:
```cuda
const float origin = (axis == 0) ? r.origin.x : (axis == 1) ? r.origin.y : r.origin.z;
```
This generates predicated instructions but the compiler may not unroll it optimally. Manually unrolling the three axes eliminates the loop and the conditionals entirely, giving the compiler a cleaner view of the computation and likely reducing both register usage and instruction count.

**`hit_scene` reloads `scene.bvh_nodes[node_index]` every iteration.** The BVHNode is fetched from global memory each loop iteration. If your BVHNode struct is small enough, prefetching fields into local variables and using `__ldg()` for read-only global memory loads would improve cache utilization since the BVH is read-only during traversal.

## Random Sampling

**`random_in_unit_sphere` uses a rejection loop.** This is a while loop with unpredictable iteration count, which is a source of warp divergence — threads in the same warp may need different numbers of iterations before finding a valid sample. A spherical coordinates approach or a Marsaglia method gives you a guaranteed single-pass sample with no rejection. Similarly `random_in_unit_disk` has the same issue.

## Numerical / Mathematical

**`scatter` has dead code.** There is an unreachable `return false` after `return true`. The compiler will eliminate it but it's worth cleaning up, especially if you add materials later where the control flow actually matters.

**`random_unit_vector` calls `random_in_unit_sphere` then normalizes.** If you replace `random_in_unit_sphere` with a direct spherical sampling method you get a unit vector for free without the normalization step.

## Launch Configuration

**Your block size is 16x16 = 256 threads.** Given your register count limits you to 3 blocks per SM, you have 768 threads per SM. Experimenting with 32x8 or 8x32 block shapes doesn't change thread count but can affect memory access patterns for the framebuffer write, since 32-wide blocks align better with warp width for coalesced writes to the row-major output buffer. With 16x16 blocks the 32 threads of a warp span two rows, which means framebuffer writes are not fully coalesced.

## Architecture Level

**The samples loop is the main barrier to wavefront restructuring.** Everything else is incremental improvement within the megakernel model. If you move toward wavefront as discussed, most of these lower-level improvements become easier to implement naturally as you restructure, since each kernel is simpler and the compiler has less state to manage simultaneously.