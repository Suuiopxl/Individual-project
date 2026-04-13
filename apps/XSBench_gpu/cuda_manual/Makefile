#===============================================================================
# XSBench CUDA Makefile
# Manually converted from openmp-threading/Makefile
#
# Build: make
# Clean: make clean
# Run:   ./XSBench_gpu -m event -s small
#===============================================================================

#===============================================================================
# User Options
#===============================================================================
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
SM          = 89

#===============================================================================
# Compilers
#===============================================================================
NVCC        = nvcc
CC          = gcc

#===============================================================================
# Program name & source code list
#===============================================================================
program = XSBench_gpu

# CUDA source files (compiled with nvcc)
cuda_source = \
	Main.cu \
	Simulation.cu \
	CudaInit.cu

# C source files (compiled with gcc, linked by nvcc)
# These are the unmodified files from openmp-threading/
c_source = \
	io.c \
	GridInit.c \
	XSutils.c \
	Materials.c \
	HostHelpers.c

cuda_obj = $(cuda_source:.cu=.o)
c_obj    = $(c_source:.c=.o)
obj      = $(cuda_obj) $(c_obj)

#===============================================================================
# Flags
#===============================================================================

# NVCC flags for .cu files
NVCCFLAGS := -arch=sm_$(SM) -Xcompiler -Wall

# GCC flags for .c files (must be C99 compatible, no CUDA)
CFLAGS := -std=gnu99 -Wall

# Linker flags
LDFLAGS = -lm

# Debug
ifeq ($(DEBUG),yes)
  NVCCFLAGS += -g -G
  CFLAGS    += -g
  LDFLAGS   += -g
endif

# Profiling
ifeq ($(PROFILE),yes)
  NVCCFLAGS += -lineinfo
  CFLAGS    += -pg
  LDFLAGS   += -pg
endif

# Optimization
ifeq ($(OPTIMIZE),yes)
  NVCCFLAGS += -O3
  CFLAGS    += -O3
endif

# Tell C files we are NOT using OpenMP (prevent #include <omp.h> errors)
# The header uses #ifdef OPENMP guards already
CFLAGS += -U OPENMP

#===============================================================================
# Targets
#===============================================================================

$(program): $(obj)
	$(NVCC) $(NVCCFLAGS) $(obj) -o $@ $(LDFLAGS)

# Compile CUDA files with nvcc
# Note: nvcc needs the header as .cuh, but we include it as XSbench_header.cuh
%.o: %.cu XSbench_header.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C files with gcc
# These files #include "XSbench_header.h" — we create a shim header below
%.o: %.c XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run:
	./$(program) -m event -s small

run_large:
	./$(program) -m event -s large

.PHONY: clean run run_large
