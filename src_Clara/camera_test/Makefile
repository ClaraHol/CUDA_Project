# Makefile for CUDA ray tracer (Freeglut + GLEW + CUDA/GL interop)
#
# Usage:
#   make          — build
#   make run      — build and launch under vglrun
#   make clean    — remove build artifacts

NVCC     := nvcc
TARGET   := raytracer
SRCS     := main.cpp raytracer.cu

# pkg-config handles include and lib paths for GLEW and Freeglut
GL_CFLAGS := $(shell pkg-config --cflags glew glut)
GL_LIBS   := $(shell pkg-config --libs   glew glut)

# Adjust -arch to match your cluster's GPU, e.g. sm_70 (V100), sm_80 (A100),
# sm_86 (A30/A40), sm_90 (H100). Use `nvidia-smi` to check.
CUDA_ARCH := -arch=sm_75

NVCC_FLAGS := $(CUDA_ARCH) -O2 -std=c++14 $(GL_CFLAGS)
LDFLAGS    := $(GL_LIBS) -lcuda

# ---------------------------------------------------------------------------
all: $(TARGET)

$(TARGET): $(SRCS) raytracer.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS) $(LDFLAGS)

run: $(TARGET)
	vglrun ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
