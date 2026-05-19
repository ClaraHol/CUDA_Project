PROJECT := raytrace
BUILD_DIR := build
.RECIPEPREFIX := >

CUDA_TARGET := $(BUILD_DIR)/$(PROJECT)

NVCC ?= nvcc

OPT ?= -O3
DBG ?= -g
ARCH ?= sm_90

CPP_STD := -std=c++17

NVCC_FLAGS := $(CPP_STD) -arch=$(ARCH) -lineinfo -Xptxas=-v -DUSE_CUDA

# Default values for scene and samples (can be overridden by "make run scene samples")
SCENE ?= cover
SAMPLES ?= 10

.PHONY: all run clean

SRC := cuda/main.cu cuda/renderer.cu cuda/kernels.cu

all: $(CUDA_TARGET)

$(CUDA_TARGET): $(SRC)
>@mkdir -p $(BUILD_DIR)
>$(NVCC) $(NVCC_FLAGS) $(OPT) $(DBG) -I cuda -o $(CUDA_TARGET) $(SRC)

# Allow: make run simple 50
ifneq ($(filter run,$(MAKECMDGOALS)),)
  SCENE := $(word 2,$(MAKECMDGOALS))
  SAMPLES := $(word 3,$(MAKECMDGOALS))
  ifeq ($(SCENE),)
    SCENE := cover
  endif
  ifeq ($(SAMPLES),)
    SAMPLES := 50
  endif
  $(eval $(SCENE):;)
  $(eval $(SAMPLES):;)
endif

run: $(CUDA_TARGET)
>./$(CUDA_TARGET) $(SCENE) $(SAMPLES)

clean:
>rm -rf $(BUILD_DIR) images/*_cuda.png
