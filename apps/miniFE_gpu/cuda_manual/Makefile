#-----------------------------------------------------------------------
# Makefile for miniFE CUDA CG benchmark
#
# Location: apps/miniFE_gpu/cuda_manual/
# References: apps/miniFE/openmp/{src,utils,fem,basic}
#-----------------------------------------------------------------------

# ===== Paths (relative to apps/miniFE_gpu/cuda_manual/) =====
MINIFE_SRC   = ../../miniFE/openmp/src
MINIFE_UTILS = ../../miniFE/openmp/utils
MINIFE_FEM   = ../../miniFE/openmp/fem
MINIFE_BASIC = ../../miniFE/openmp/basic

# ===== Compilers =====
NVCC = nvcc
MPICXX = mpicxx

# Detect GPU compute capability
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = 89
endif

# ===== Flags =====
MINIFE_TYPES = \
    -DMINIFE_SCALAR=double \
    -DMINIFE_LOCAL_ORDINAL=int \
    -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_DEFS = \
    $(MINIFE_TYPES) \
    -DMINIFE_CSR_MATRIX \
    -DHAVE_MPI \
    -DMPICH_IGNORE_CXX_SEEK \
    -DMINIFE_REPORT_RUSAGE

INCLUDES = \
    -I. \
    -I$(MINIFE_SRC) \
    -I$(MINIFE_UTILS) \
    -I$(MINIFE_FEM) \
    -I$(MINIFE_BASIC)

MPI_CFLAGS  = $(shell $(MPICXX) -showme:compile 2>/dev/null || echo "")
MPI_LDFLAGS = $(shell $(MPICXX) -showme:link 2>/dev/null || echo "-lmpi -lmpi_cxx")

NVCCFLAGS = -O2 -arch=sm_$(CUDA_ARCH) -std=c++14 \
            -Xcompiler "-O2 -fopenmp" \
            $(INCLUDES) $(MINIFE_DEFS) $(MPI_CFLAGS)

LDFLAGS = $(MPI_LDFLAGS) -lgomp

# ===== Object files =====
UTIL_OBJS = BoxPartition.o utils.o mytimer.o param_utils.o YAML_Doc.o YAML_Element.o
CUDA_OBJS = miniFE_cuda_kernels.o miniFE_cuda_main.o
ALL_OBJS  = $(UTIL_OBJS) $(CUDA_OBJS)

TARGET = miniFE_cuda_bench
NX ?= 128

# ===== Rules =====
all: common_files $(TARGET)

$(TARGET): $(ALL_OBJS)
	$(NVCC) -arch=sm_$(CUDA_ARCH) $(ALL_OBJS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Build successful: $(TARGET)"
	@echo "Run: mpirun -np 1 ./$(TARGET) -nx $(NX) -ny $(NX) -nz $(NX)"

miniFE_cuda_kernels.o: miniFE_cuda_kernels.cu miniFE_cuda_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

miniFE_cuda_main.o: miniFE_cuda_main.cu miniFE_cuda_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

BoxPartition.o: $(MINIFE_UTILS)/BoxPartition.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

utils.o: $(MINIFE_UTILS)/utils.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

mytimer.o: $(MINIFE_UTILS)/mytimer.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

param_utils.o: $(MINIFE_UTILS)/param_utils.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

YAML_Doc.o: $(MINIFE_SRC)/YAML_Doc.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

YAML_Element.o: $(MINIFE_SRC)/YAML_Element.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

common_files:
	@if [ -f "$(MINIFE_SRC)/get_common_files" ]; then \
		cd $(MINIFE_SRC) && bash get_common_files; \
	fi
	@if [ -f "$(MINIFE_SRC)/generate_info_header" ]; then \
		cd $(MINIFE_SRC) && bash generate_info_header "nvcc" "$(NVCCFLAGS)" "miniFE" "MINIFE" 2>/dev/null || true; \
	fi

run: all
	OMP_NUM_THREADS=1 mpirun --oversubscribe --allow-run-as-root -np 1 \
		./$(TARGET) -nx $(NX) -ny $(NX) -nz $(NX)

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean run common_files
