OPT	= -g -O3
ISA	= 
PARA	= -fopenmp

# Compiler and flags
CXX	= nvc++ -cuda -mp=gpu -Minfo --diag_suppress integer_sign_change,cuda_compile
CXXFLAGS= $(OPT) $(ISA) $(PARA)

NVCC	= nvcc
CUFLAGS= --compiler-options "$(OPT) $(PARA)" -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 68


# Target executable
TARGET = main

# For deleting the target
TARGET_DEL = main.exe

# Source files
SRCS = main.cu
 
# Object files
OBJS = $(SRCS:.cu=.o)

# Default rule to build and run the executable
all: $(TARGET) run

# Rule to link object files into the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

.SUFFIXES: .cu .c
.cu.o:
	$(NVCC) -o $*.o -c $*.cu $(CUFLAGS) $(SOFLAGS) $(INCLUDES)

# Rule to compile .cpp files into .o files
.cpp.o:
	$(CXX) -o $*.o -c $*.cpp $(OPT) $(PARA) $(SOFLAGS) $(INCLUDES)

# Rule to run the executable
run: $(TARGET)
	./$(TARGET) > cuda_image.ppm


clean:
	@/bin/rm -f *.o core

realclean: clean
	@rm -f $(TARGET)



