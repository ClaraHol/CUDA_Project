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

# Default values for scene and samples.
SCENE ?= cover
SAMPLES ?= 50

.PHONY: all run clean

SRC := cuda/main.cu cuda/renderer.cu cuda/kernels.cu cuda/bvh_builder.cpp

all: $(CUDA_TARGET)

$(CUDA_TARGET): $(SRC)
>@mkdir -p $(BUILD_DIR)
>$(NVCC) $(NVCC_FLAGS) $(OPT) $(DBG) -I cuda -o $(CUDA_TARGET) $(SRC)

RUN_ARGS := $(filter-out run,$(MAKECMDGOALS))
ifneq ($(filter run,$(MAKECMDGOALS)),)
  ifneq ($(strip $(RUN_ARGS)),)
    FIRST_ARG := $(word 1,$(RUN_ARGS))
    SECOND_ARG := $(word 2,$(RUN_ARGS))
    # Positional run args are forwarded as: scene, samples, max_depth.
    MAX_DEPTH := $(word 3,$(RUN_ARGS))

    ifeq ($(FIRST_ARG),simple)
      SCENE := simple
      ifneq ($(strip $(SECOND_ARG)),)
        SAMPLES := $(SECOND_ARG)
      endif
    else ifeq ($(FIRST_ARG),cover)
      SCENE := cover
      ifneq ($(strip $(SECOND_ARG)),)
        SAMPLES := $(SECOND_ARG)
      endif
    else ifeq ($(FIRST_ARG),spiral)
      SCENE := spiral
      ifneq ($(strip $(SECOND_ARG)),)
        SAMPLES := $(SECOND_ARG)
      endif
    else
      SCENE := $(FIRST_ARG)
      ifneq ($(strip $(SECOND_ARG)),)
        SAMPLES := $(SECOND_ARG)
      else
        SAMPLES := $(FIRST_ARG)
        SCENE := cover
      endif
    endif

    $(RUN_ARGS):;
  endif
endif

run: $(CUDA_TARGET)
>./$(CUDA_TARGET) $(SCENE) $(SAMPLES) $(MAX_DEPTH)

clean:
>rm -rf $(BUILD_DIR) images/*_cuda.png
