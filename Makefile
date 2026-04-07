PROJECT := raytrace
BUILD_DIR := build

CPU_TARGET := $(BUILD_DIR)/$(PROJECT)_cpu
CUDA_TARGET := $(BUILD_DIR)/$(PROJECT)_cuda

CPU_SRC := src/cpu/main.cpp
CUDA_SRC := src/cuda/cuda_renderer.cu src/cuda/cuda_kernels.cu

CXX ?= g++
NVCC ?= nvcc

OPT ?= -O3
DBG ?= -g
PARA ?= -fopenmp
ARCH ?= sm_90

CPP_STD := -std=c++17

CPU_CXXFLAGS := $(CPP_STD) $(DBG) $(OPT) $(PARA)
NVCC_CPUFLAGS := --compiler-options "$(DBG) $(OPT) $(PARA)"
NVCC_FLAGS := -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-v -DUSE_CUDA

.PHONY: all cpu cuda run-cpu run-omp run-cuda run-all clean

# One-command default for HPC GPU build
all: cuda

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_SRC) | $(BUILD_DIR)
	$(CXX) $(CPU_CXXFLAGS) $< -o $@

cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CPU_SRC) $(CUDA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_CPUFLAGS) $(CPU_SRC) $(CUDA_SRC) -o $@

run-cpu: cpu
	./$(CPU_TARGET) --mode cpu

run-omp: cpu
	./$(CPU_TARGET) --mode omp

run-cuda: cuda
	./$(CUDA_TARGET) --mode cuda

run-all: cuda
	./$(CUDA_TARGET) --mode all

clean:
	rm -rf $(BUILD_DIR)
