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

# Default values for scene, mode, and samples
SCENE ?= cover
MODE ?= cuda
SAMPLES ?= 10

.PHONY: all cpu cuda clean run

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

# Value-based argument parsing: handles (mode) cpu/omp/cuda/all, (scene) simple/cover, (samples) integer
# Usage: make run simple cuda 100
#        make run 50 cover omp
#        make run cuda                (uses defaults for scene and samples)
run: cuda
	@PARSED_SCENE=$(SCENE); \
	PARSED_MODE=$(MODE); \
	PARSED_SAMPLES=$(SAMPLES); \
	for arg in $(filter-out $@,$(MAKECMDGOALS)); do \
	  case $$arg in \
	    simple) PARSED_SCENE=simple ;; \
	    cover) PARSED_SCENE=cover ;; \
	    cpu|omp|cuda|all) PARSED_MODE=$$arg ;; \
	    *) PARSED_SAMPLES=$$arg ;; \
	  esac; \
	done; \
	echo "Running: scene $$PARSED_SCENE mode $$PARSED_MODE samples $$PARSED_SAMPLES"; \
	./$(CUDA_TARGET) scene $$PARSED_SCENE mode $$PARSED_MODE samples $$PARSED_SAMPLES

# Prevent Make from treating parsed arguments as targets
simple cover cpu omp cuda all:
	@:

# Catch-all for numeric arguments (samples)
%:
	@:

clean:
	rm -rf $(BUILD_DIR)
