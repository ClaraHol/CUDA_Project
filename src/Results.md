# Benchmark Results

Five runs were recorded for each renderer mode on the compute node.
(I forgot to note the exact input parameters, so these will need to be redone.)

## Summary

### CPU
- Run 1: 3.48244
- Run 2: 3.49278
- Run 3: 3.53821
- Run 4: 3.51690
- Run 5: 3.49410
- Average: 3.504886

### OMP
- Run 1: 4.69081
- Run 2: 4.68537
- Run 3: 4.71072
- Run 4: 4.71026
- Run 5: 4.65786
- Average: 4.691004

### CUDA
- Run 1: 0.0204246
- Run 2: 0.0204268
- Run 3: 0.0204235
- Run 4: 0.0204215
- Run 5: 0.0204235
- Average: 0.020424

### Benchmark - cover, 100 samples, from version - https://github.com/ClaraHol/CUDA_Project/releases/tag/initial-version
cpu - 334.42 seconds
omp - 27.31 seconds - speed up 12.4x
cuda - 0.6802 seconds - speed up 491x
## 10 samples
cpu - 34.4226
cpu - 35.9947
cpu - 33.4329
cpu - 34.2932
avg - 34.5359
## 50 samples
cpu - 175.889

## Projected time for 500 samples
cpu - 1726.8
## Projected time for 1000 samples
cpu - 3453.6
## 500 samples 
cuda - 3.35926
cuda - 3.4615
cuda - 3.41781
## 1000 samples
cuda - 6.89401
cuda - 6.77574
cuda - 6.81927
cuda - 6.86477


### Benchmark - cover, 500 samples, maxdepth=2
cuda - 0.4867 seconds
### Benchmark - cover, 500 samples, maxdepth=2
cuda - 4.87055 seconds

