import matplotlib.pyplot as plt



num_threads = [1, 2, 4, 8, 16, 32, 64, 72]
times_simple = [3.20085, 4.38779, 4.52852, 4.46478, 4.22346, 4.14732, 4.13781, 4.1944]
times_complex = [1560.14, 944.241, 562.236, 294.104, 176.45, 113.402, 98.6726, 98.7433]

speed_up = [times_simple[0]/times_simple[i] for i in range(len(times_simple))]
speed_up_complex = [times_complex[0]/times_complex[i] for i in range(len(times_complex))]

plt.figure()
#plt.plot(num_threads, speed_up, label = "Simple scene", marker = "x")
plt.plot(num_threads, speed_up_complex, label = "Test scene",  marker = "o")
plt.plot(num_threads, num_threads, label = "Ideal", marker = "d")
plt.xlabel("Number of threads")
plt.ylabel("Speed up")
plt.legend()
plt.title("Speed up of parallelized ray tracer with openMP")
plt.savefig("OpenMP_Speed_up")


## BVH Timings
times_complex = [404.999, 284.303, 174.301, 114.498, 100.765, 92.9047, 96.1448, 98.3623]
speed_up_complex = [times_complex[0]/times_complex[i] for i in range(len(times_complex))]

plt.figure()

plt.plot(num_threads, speed_up_complex, label = "Complex scene",  marker = "o")
plt.plot(num_threads, num_threads, label = "Ideal", marker = "d")
plt.xlabel("Number of threads")
plt.ylabel("Speed up")
plt.legend()
plt.title("Speed up of parallelized BVH ray tracer with openMP")
plt.savefig("BVH_OpenMP_Speed_up")


# Scaling with max depth
max_depths = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

times_linear = [0.676392, 3.65448, 4.52722, 5.08132, 5.61507, 6.12502, 6.58561, 7.12842, 7.46626, 7.76187, 8.00302]
times_bvh = [20.5315, 177.296, 215.316, 254.09, 295.864, 335.931, 398.408, 432.144, 453.096, 407.627, 430.123]
times_wf = [134.412, 378.349, 437.1, 474.673, 506.785, 536.381, 564.908, 591.614, 616.894, 640.769, 664.552]

speed_up_linear = [times_linear[0]/times_linear[i]*max_depths[i] for i in range(len(times_linear))]
speed_up_bvh = [times_bvh[0]/times_bvh[i]*max_depths[i] for i in range(len(times_bvh))]
speed_up_wf = [times_wf[0]/times_wf[i]*max_depths[i] for i in range(len(times_wf))]
plt.figure()
plt.plot(max_depths, speed_up_linear, label = "Linear", marker = "d")
plt.plot(max_depths, speed_up_bvh, label = "BVH", marker = "x")
plt.plot(max_depths, speed_up_wf, label = "Wavefront",  marker = "o")
plt.xlabel("Max depth")
plt.ylabel("Relative Speed up")
plt.legend()
plt.title("Impact of maximum bounce depth")
plt.tight_layout()
plt.savefig("BVH_wf_max_depth")


# Scaling with Samples per pixel
spp = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
times_linear = [0.0374896, 1.84856, 3.66673, 5.5129, 7.31299, 9.23033, 10.949, 12.8748, 15.158, 16.6323, 19.3946, 20.0935, 22.3298, 24.0927, 26.3435, 27.4124]
times_bvh = [1.92857, 90.3566, 179.472,  272.519, 356.418, 449.691, 536.747, 623.797, 720.818,  792.625, 893.505, 973.957, 1053.49, 1154.15, 1231.92, 1330.63]
times_wf = [ 5.03997, 192.182, 385.971, 579.246, 773.967, 969.532, 1164.75, 1360.66, 1560.29, 1729.2, 1953.17, 2116.77, 2318.88, 2512.84, 2714.87, 2911.71]


speed_up_linear = [times_linear[2]/times_linear[i]*(spp[i]/spp[2]) for i in range(2, len(times_linear))]
speed_up_bvh = [times_bvh[2]/times_bvh[i]*(spp[i]/spp[2]) for i in range(2, len(times_bvh))]
speed_up_wf = [times_wf[2]/times_wf[i]*(spp[i]/spp[2]) for i in range(2, len(times_wf))]
plt.figure()
plt.plot(spp[2:], speed_up_linear, label = "Linear", marker = "d")
plt.plot(spp[2:], speed_up_bvh, label = "BVH", marker = "x")
plt.plot(spp[2:], speed_up_wf, label = "Wavefront",  marker = "o")
plt.xlabel("Samples per pixel")
plt.ylabel("Relative Speed Up")
plt.legend()
plt.title("Impact of number of samples per pixel")
plt.tight_layout()
plt.savefig("BVH_wf_spp")