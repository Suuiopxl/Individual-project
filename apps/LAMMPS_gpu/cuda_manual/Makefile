NVCC = nvcc

CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),)
    CUDA_ARCH = 89
endif

NVCCFLAGS = -O2 -arch=sm_$(CUDA_ARCH) -std=c++14 \
            -Xcompiler "-O2"

TARGET = lj_cuda_bench

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
