#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# setup_and_build_cuda.sh — Prepare and build XSBench CUDA version
#
# Usage:
#   cd ~/Individual-project
#   bash apps/XSBench_gpu/setup_and_build_cuda.sh
#
# This script:
#   1. Copies unmodified .c files from openmp-threading/ into XSBench_gpu/
#   2. Verifies all required files are present
#   3. Builds the CUDA version
#   4. Runs a quick validation test
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$SCRIPT_DIR"
CPU_DIR="$(cd "$SCRIPT_DIR/../../XSBench/openmp-threading" && pwd)"

echo "=============================================="
echo "  XSBench CUDA Build Setup"
echo "  GPU dir: $GPU_DIR"
echo "  CPU dir: $CPU_DIR"
echo "=============================================="

# Step 1: Copy unmodified C source files from CPU version
echo ""
echo "[1/4] Copying C source files from openmp-threading/..."

for f in io.c GridInit.c XSutils.c Materials.c; do
    if [ -f "$CPU_DIR/$f" ]; then
        cp "$CPU_DIR/$f" "$GPU_DIR/$f"
        echo "  Copied: $f"
    else
        echo "  ERROR: $CPU_DIR/$f not found!"
        exit 1
    fi
done

# Step 2: Verify all required files
echo ""
echo "[2/4] Verifying file structure..."

REQUIRED_FILES=(
    "Main.cu"
    "Simulation.cu"
    "CudaInit.cu"
    "XSbench_header.cuh"
    "XSbench_header.h"
    "Makefile"
    "io.c"
    "GridInit.c"
    "XSutils.c"
    "Materials.c"
)

ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$GPU_DIR/$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Some files are missing. Cannot proceed."
    exit 1
fi

# Step 3: Build
echo ""
echo "[3/4] Building CUDA version..."
cd "$GPU_DIR"
make clean 2>/dev/null || true
make 2>&1 | tee build_cuda.log

if [ ! -f "XSBench_gpu" ]; then
    echo ""
    echo "ERROR: Build failed. Check build_cuda.log"
    tail -20 build_cuda.log
    exit 1
fi

echo ""
echo "  Build successful: $(ls -lh XSBench_gpu | awk '{print $5, $9}')"

# Step 4: Quick validation run
echo ""
echo "[4/4] Running quick validation (-m event -s small -l 100000)..."
./XSBench_gpu -m event -s small -l 100000 2>&1 | tail -20

echo ""
echo "=============================================="
echo "  XSBench CUDA build complete!"
echo "=============================================="
echo ""
echo "  Run benchmarks:"
echo "    ./XSBench_gpu -m event -s small     # ~17M lookups"
echo "    ./XSBench_gpu -m event -s large     # larger problem"
echo ""
echo "  Compare against CPU baseline:"
echo "    cd ~/Individual-project/apps/XSBench/openmp-threading"
echo "    make clean && make"
echo "    ./XSBench -m event -s small -t 4"
echo ""
