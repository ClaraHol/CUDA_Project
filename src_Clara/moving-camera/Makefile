

#OPT	= -g -O3 $(GL_CFLAGS)
#ISA	= 
#PARA	= -fopenmp  -use_fast_math $(GL_LIBS) 

# Compiler and flags
#CXX	= nvc++ -cuda -mp=gpu -Minfo --diag_suppress integer_sign_change,cuda_compile 
#CXXFLAGS= $(OPT) $(ISA) $(PARA)

# Use sm_90 for H100 and sm_25 for VL_graphics node
#NVCC	= nvcc
#CUFLAGS= --compiler-options "$(OPT) $(PARA)" -Xptxas=-v -arch=sm_75 -lineinfo -diag-suppress 68 


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

NVCC_FLAGS := $(CUDA_ARCH) -O3 -std=c++14 $(GL_CFLAGS) -lineinfo -diag-suppress 68 
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



# Target executable
#TARGET = main

# For deleting the target
#TARGET_DEL = main.exe

# Source files
#SRCS = main.c++ raytracer.cu
 
# Object files
#OBJS = $(SRCS:.cu=.o)

# Default rule to build and run the executable
#all: $(TARGET) run

# Rule to link object files into the target executable
#$(TARGET): $(OBJS)
#	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

#.SUFFIXES: .cu .c
#.cu.o:
#	$(NVCC) -o $*.o -c $*.cu $(CUFLAGS) $(SOFLAGS) $(INCLUDES)


#stb_impl.o: stb_impl.cpp
#	g++ -c stb_impl.cpp -o stb_impl.o -O3
# Rule to compile .cpp files into .o files
#.cpp.o:
#	$(CXX) -o $*.o -c $*.cpp $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

# Rule to run the executable
#run: $(TARGET)
#	vlgrun ./$(TARGET) 


#clean:
#	@/bin/rm -f *.o core

#realclean: clean
#	@rm -f $(TARGET)



