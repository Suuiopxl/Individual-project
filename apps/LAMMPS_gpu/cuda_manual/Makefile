#-----------------------------------------------------------------------
# Makefile for LAMMPS LJ CUDA force benchmark
#
# Location: apps/LAMMPS_gpu/cuda_manual/
# Reference: apps/LAMMPS_gpu/src/pair_lj_cut.cpp (CPU baseline)
#
# Usage:
#   make                  # build with auto-detected GPU arch
#   make CUDA_ARCH=89     # force specific arch (e.g. sm_89 for RTX 4080)
#   make clean
#   ./lj_cuda_bench       # run with default 32000 atoms
#   ./lj_cuda_bench 30    # run with 30^3*4 = 108000 atoms
#-----------------------------------------------------------------------

NVCC = nvcc

# Auto-detect GPU compute capability
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = 89
endif

NVCCFLAGS = -O2 -arch=sm_$(CUDA_ARCH) -std=c++14 \
            -Xcompiler "-O2"

TARGET = lj_cuda_bench

# Object files
OBJS = lj_force_cuda.o lj_bench_main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -arch=sm_$(CUDA_ARCH) $(OBJS) -o $@
	@echo ""
	@echo "Build successful: $(TARGET)"
	@echo "Run:   ./$(TARGET)        (default 32000 atoms)"
	@echo "       ./$(TARGET) 30     (108000 atoms)"
	@echo "       ./$(TARGET) 40     (256000 atoms)"

lj_force_cuda.o: lj_force_cuda.cu lj_force_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

lj_bench_main.o: lj_bench_main.cu lj_force_cuda.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
