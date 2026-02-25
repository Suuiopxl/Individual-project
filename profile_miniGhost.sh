#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# profile_miniGhost.sh
#
# 功能：
#   Phase 0 - 安装依赖 & 编译 miniGhost（gprof 版 + 普通版）
#   Phase 1 - Baseline 基准测试（多次运行取平均）
#   Phase 2 - gprof 性能分析（函数级 flat profile）
#   Phase 3 - perf 性能分析（硬件计数器 + 热点函数）
#   Phase 4 - (可选) Intel VTune 性能分析
#   Phase 5 - 汇总：自动提取 Top-N 瓶颈函数，输出 JSON 供 LLM 消费
#
# 用法：
#   chmod +x profile_miniGhost.sh
#   ./profile_miniGhost.sh              # 默认 2 MPI ranks
#   NP=4 ./profile_miniGhost.sh         # 4 MPI ranks
#   ENABLE_VTUNE=1 ./profile_miniGhost.sh  # 同时启用 VTune
#
# 环境：GitHub Codespace / Ubuntu 22.04+ / 可迁移至 HPC
###############################################################################

# ======================== 配置区 ========================
NP="${NP:-2}"                          # MPI 进程数
BASELINE_RUNS="${BASELINE_RUNS:-3}"    # 基准测试重复次数
TOP_N="${TOP_N:-10}"                   # 报告中提取的 Top-N 热点函数数
ENABLE_VTUNE="${ENABLE_VTUNE:-0}"      # 是否启用 VTune（默认关闭）

WORK_DIR="$HOME/miniGhost_profiling"
SRC_DIR="$HOME/miniGhost/ref"
REPORT_DIR="$WORK_DIR/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# miniGhost 问题规模参数（可调大以获得更有意义的 profiling 数据）
# 默认值适中，运行约 10-30 秒
MG_ARGS="--nx 100 --ny 100 --nz 100 --num_tsteps 20 --num_vars 5"

mkdir -p "$WORK_DIR" "$REPORT_DIR"

# ======================== 辅助函数 ========================
log()   { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn()  { echo -e "\n\033[1;33m[WARN] $*\033[0m"; }
err()   { echo -e "\n\033[1;31m[ERROR] $*\033[0m" >&2; }

check_cmd() {
  command -v "$1" &>/dev/null || { err "$1 not found. Please install it."; return 1; }
}

# ======================== Phase 0: 依赖 & 编译 ========================
phase0_build() {
  log "Phase 0: 安装依赖 & 编译 miniGhost"

  # --- 0.1 系统依赖 ---
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    git make build-essential gfortran \
    openmpi-bin libopenmpi-dev \
    linux-tools-common linux-tools-generic \
    binutils 2>/dev/null || true

  # perf 可能需要特殊处理（Codespace 内核版本）
  # 尝试获取与内核匹配的 linux-tools
  KVER=$(uname -r)
  sudo apt-get install -y -qq "linux-tools-${KVER}" 2>/dev/null || \
    warn "无法安装 linux-tools-${KVER}，perf 可能不可用（Codespace 限制）"

  # --- 0.2 获取源码（如果尚未存在） ---
  if [ ! -d "$SRC_DIR" ]; then
    log "克隆 miniGhost 源码..."
    cd "$HOME"
    rm -rf miniGhost
    git clone https://github.com/Mantevo/miniGhost.git
    cd miniGhost/ref
  else
    cd "$SRC_DIR"
  fi

  # --- 0.3 修复 makefile ---
  if [ ! -f makefile.mpi.gnu ]; then
    err "makefile.mpi.gnu not found in $SRC_DIR"
    exit 1
  fi

  sed -i 's|^MPI_LOC *=.*$|MPI_LOC =|g'     makefile.mpi.gnu
  sed -i 's|^MPI_INCLUDE *=.*$|MPI_INCLUDE =|g' makefile.mpi.gnu

  for flag in '-ffree-form' '-fallow-argument-mismatch' '-ffree-line-length-none'; do
    grep -q "FFLAGS += $flag" makefile.mpi.gnu || echo "FFLAGS += $flag" >> makefile.mpi.gnu
  done

  # --- 0.4 编译普通版本（用于 baseline + perf） ---
  log "编译 miniGhost（普通版，-O2）..."
  make -f makefile.mpi.gnu clean >/dev/null 2>&1 || true
  make -f makefile.mpi.gnu CC=mpicc FC=mpifort LD=mpifort \
    > "$REPORT_DIR/build_normal.log" 2>&1 || {
    err "普通版编译失败，查看 $REPORT_DIR/build_normal.log"
    tail -30 "$REPORT_DIR/build_normal.log"
    exit 1
  }
  cp miniGhost.x "$WORK_DIR/miniGhost_normal.x"

  # --- 0.5 编译 gprof 版本（加 -pg） ---
  log "编译 miniGhost（gprof 版，-pg）..."
  make -f makefile.mpi.gnu clean >/dev/null 2>&1 || true

  # 创建临时 makefile，注入 -pg
  cp makefile.mpi.gnu makefile.mpi.gnu.gprof
  sed -i 's/^FFLAGS *=/& -pg /' makefile.mpi.gnu.gprof
  sed -i 's/^CFLAGS *=/& -pg /' makefile.mpi.gnu.gprof 2>/dev/null || true
  # 确保链接时也带 -pg
  echo 'LDFLAGS += -pg' >> makefile.mpi.gnu.gprof

  make -f makefile.mpi.gnu.gprof CC=mpicc FC=mpifort LD=mpifort \
    > "$REPORT_DIR/build_gprof.log" 2>&1 || {
    err "gprof 版编译失败，查看 $REPORT_DIR/build_gprof.log"
    tail -30 "$REPORT_DIR/build_gprof.log"
    exit 1
  }
  cp miniGhost.x "$WORK_DIR/miniGhost_gprof.x"

  log "编译完成 ✓"
  ls -lh "$WORK_DIR"/miniGhost_*.x
}

# ======================== Phase 1: Baseline 基准测试 ========================
phase1_baseline() {
  log "Phase 1: Baseline 基准测试（${BASELINE_RUNS} 次，${NP} ranks）"

  cd "$WORK_DIR"
  local times=()
  local baseline_log="$REPORT_DIR/baseline_${TIMESTAMP}.log"

  for i in $(seq 1 "$BASELINE_RUNS"); do
    log "  Baseline run $i / $BASELINE_RUNS"

    # 用 time 获取 wall-clock
    local t0
    t0=$(date +%s%N)

    mpirun --oversubscribe --allow-run-as-root -np "$NP" \
      ./miniGhost_normal.x $MG_ARGS \
      >> "$baseline_log" 2>&1

    local t1
    t1=$(date +%s%N)
    local elapsed_ms=$(( (t1 - t0) / 1000000 ))
    times+=("$elapsed_ms")
    echo "  Run $i: ${elapsed_ms} ms"
  done

  # 计算统计
  local sum=0
  for t in "${times[@]}"; do sum=$((sum + t)); done
  local avg=$((sum / BASELINE_RUNS))

  local min=${times[0]} max=${times[0]}
  for t in "${times[@]}"; do
    (( t < min )) && min=$t
    (( t > max )) && max=$t
  done

  cat > "$REPORT_DIR/baseline_summary_${TIMESTAMP}.txt" <<EOF
==============================
miniGhost Baseline Summary
==============================
Date:       $(date)
MPI ranks:  $NP
Runs:       $BASELINE_RUNS
Args:       $MG_ARGS

Wall-clock times (ms): ${times[*]}
Average:  ${avg} ms
Min:      ${min} ms
Max:      ${max} ms
==============================
EOF

  cat "$REPORT_DIR/baseline_summary_${TIMESTAMP}.txt"
  log "Baseline 完成 ✓  详细日志: $baseline_log"
}

# ======================== Phase 2: gprof 性能分析 ========================
phase2_gprof() {
  log "Phase 2: gprof 性能分析"

  cd "$WORK_DIR"
  # gprof 需要单进程运行（多进程下 gmon.out 会被覆盖）
  # 方案：用 1 个 rank 运行，或者重命名 gmon.out
  # 这里用单 rank 获得干净的 profile

  log "  运行 gprof 版（单 rank）..."
  rm -f gmon.out
  mpirun --oversubscribe --allow-run-as-root -np 1 \
    ./miniGhost_gprof.x $MG_ARGS \
    > "$REPORT_DIR/gprof_run_${TIMESTAMP}.log" 2>&1

  if [ ! -f gmon.out ]; then
    warn "gmon.out 未生成，跳过 gprof 分析"
    return
  fi

  # 生成 flat profile
  gprof ./miniGhost_gprof.x gmon.out \
    > "$REPORT_DIR/gprof_full_${TIMESTAMP}.txt" 2>&1

  # 提取 flat profile 的前 N 行热点函数
  log "  提取 Top-$TOP_N 热点函数（gprof flat profile）..."

  # flat profile 以 "  %   cumulative   self" 开头
  awk '
    /^  %   cumulative   self/ { start=1; next }
    /^$/ && start { exit }
    start && NR>0 { print }
  ' "$REPORT_DIR/gprof_full_${TIMESTAMP}.txt" \
    | head -n "$TOP_N" \
    > "$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"

  log "gprof 分析完成 ✓"
  echo "--- gprof Top-$TOP_N 热点函数 ---"
  cat "$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"
  echo ""
}

# ======================== Phase 3: perf 性能分析 ========================
phase3_perf() {
  log "Phase 3: perf 性能分析"

  cd "$WORK_DIR"

  # 检测 perf 可用性
  if ! command -v perf &>/dev/null; then
    warn "perf 不可用，跳过 Phase 3"
    warn "在 HPC 上通常可用；Codespace 中可能受限"
    return
  fi

  # 3.1 perf stat - 硬件计数器概览
  log "  perf stat: 收集硬件计数器..."
  perf stat -o "$REPORT_DIR/perf_stat_${TIMESTAMP}.txt" \
    mpirun --oversubscribe --allow-run-as-root -np "$NP" \
    ./miniGhost_normal.x $MG_ARGS \
    > "$REPORT_DIR/perf_stat_run_${TIMESTAMP}.log" 2>&1 || {
    warn "perf stat 失败（可能是 Codespace 内核限制）"
  }

  if [ -f "$REPORT_DIR/perf_stat_${TIMESTAMP}.txt" ]; then
    echo "--- perf stat 概览 ---"
    cat "$REPORT_DIR/perf_stat_${TIMESTAMP}.txt"
  fi

  # 3.2 perf record + report - 采样热点
  log "  perf record: 采样 CPU 热点..."
  perf record -g -o "$WORK_DIR/perf.data" -- \
    mpirun --oversubscribe --allow-run-as-root -np "$NP" \
    ./miniGhost_normal.x $MG_ARGS \
    > "$REPORT_DIR/perf_record_run_${TIMESTAMP}.log" 2>&1 || {
    warn "perf record 失败"
    return
  }

  # 生成文本报告
  perf report -i "$WORK_DIR/perf.data" --stdio --no-children \
    > "$REPORT_DIR/perf_report_${TIMESTAMP}.txt" 2>&1 || true

  # 提取 Top-N 热点
  grep -E '^\s+[0-9]+\.[0-9]+%' "$REPORT_DIR/perf_report_${TIMESTAMP}.txt" \
    | head -n "$TOP_N" \
    > "$REPORT_DIR/perf_top${TOP_N}_${TIMESTAMP}.txt" 2>/dev/null || true

  if [ -s "$REPORT_DIR/perf_top${TOP_N}_${TIMESTAMP}.txt" ]; then
    echo "--- perf Top-$TOP_N 热点函数 ---"
    cat "$REPORT_DIR/perf_top${TOP_N}_${TIMESTAMP}.txt"
  fi

  log "perf 分析完成 ✓"
}

# ======================== Phase 4: VTune（可选） ========================
phase4_vtune() {
  if [ "$ENABLE_VTUNE" != "1" ]; then
    log "Phase 4: VTune（已跳过，设置 ENABLE_VTUNE=1 启用）"
    return
  fi

  log "Phase 4: Intel VTune 性能分析"

  # --- 4.1 安装 VTune ---
  if ! command -v vtune &>/dev/null; then
    log "  安装 Intel VTune..."

    # 添加 Intel oneAPI 仓库
    wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | sudo gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg 2>/dev/null

    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] \
      https://apt.repos.intel.com/oneapi all main" \
      | sudo tee /etc/apt/sources.list.d/oneAPI.list > /dev/null

    sudo apt-get update -qq
    sudo apt-get install -y -qq intel-oneapi-vtune

    # Source VTune 环境
    VTUNE_VARS="/opt/intel/oneapi/vtune/latest/env/vars.sh"
    if [ -f "$VTUNE_VARS" ]; then
      source "$VTUNE_VARS"
    else
      err "VTune 安装后找不到 vars.sh"
      return
    fi
  fi

  # --- 4.2 VTune Hotspots 分析 ---
  local vtune_result="$WORK_DIR/vtune_hotspots_${TIMESTAMP}"

  log "  VTune hotspots 采集..."
  vtune -collect hotspots \
    -result-dir "$vtune_result" \
    -- mpirun --oversubscribe --allow-run-as-root -np "$NP" \
    ./miniGhost_normal.x $MG_ARGS \
    > "$REPORT_DIR/vtune_collect_${TIMESTAMP}.log" 2>&1 || {
    warn "VTune hotspots 采集失败"
    return
  }

  # --- 4.3 生成文本报告 ---
  vtune -report hotspots \
    -result-dir "$vtune_result" \
    -format text \
    -report-output "$REPORT_DIR/vtune_hotspots_${TIMESTAMP}.txt" \
    > /dev/null 2>&1

  vtune -report summary \
    -result-dir "$vtune_result" \
    -format text \
    -report-output "$REPORT_DIR/vtune_summary_${TIMESTAMP}.txt" \
    > /dev/null 2>&1

  # --- 4.4 生成 CSV（方便后续 LLM 解析） ---
  vtune -report hotspots \
    -result-dir "$vtune_result" \
    -format csv \
    -report-output "$REPORT_DIR/vtune_hotspots_${TIMESTAMP}.csv" \
    > /dev/null 2>&1

  echo "--- VTune Hotspots ---"
  head -30 "$REPORT_DIR/vtune_hotspots_${TIMESTAMP}.txt" 2>/dev/null || true

  log "VTune 分析完成 ✓"
}

# ======================== Phase 5: 汇总报告 & JSON 输出 ========================
phase5_summary() {
  log "Phase 5: 生成汇总报告"

  local summary_file="$REPORT_DIR/profiling_summary_${TIMESTAMP}.json"

  # 收集 gprof 热点信息，转成简单 JSON 数组
  local gprof_hotspots="[]"
  local gprof_file="$REPORT_DIR/gprof_top${TOP_N}_${TIMESTAMP}.txt"
  if [ -s "$gprof_file" ]; then
    # gprof flat profile 格式：
    # %time  cumulative  self  calls  self_ms/call  total_ms/call  name
    gprof_hotspots=$(awk '
      BEGIN { printf "[" }
      NF >= 7 {
        gsub(/"/, "\\\"", $7)
        if (NR > 1) printf ","
        printf "\n    {\"pct_time\": \"%s\", \"cumulative_sec\": \"%s\", \"self_sec\": \"%s\", \"calls\": \"%s\", \"function\": \"%s\"}", $1, $2, $3, $4, $7
      }
      END { printf "\n  ]" }
    ' "$gprof_file")
  fi

  # 收集 perf 热点信息
  local perf_hotspots="[]"
  local perf_file="$REPORT_DIR/perf_top${TOP_N}_${TIMESTAMP}.txt"
  if [ -s "$perf_file" ]; then
    perf_hotspots=$(awk '
      BEGIN { printf "[" }
      {
        # perf report 格式：  XX.XX%  command  shared_object  [.] symbol
        match($0, /([0-9]+\.[0-9]+)%/, pct)
        # 提取函数名（最后一个字段）
        func = $NF
        gsub(/"/, "\\\"", func)
        if (NR > 1) printf ","
        printf "\n    {\"overhead_pct\": \"%s\", \"function\": \"%s\"}", pct[1], func
      }
      END { printf "\n  ]" }
    ' "$perf_file")
  fi

  # 写入 JSON
  cat > "$summary_file" <<ENDJSON
{
  "application": "miniGhost",
  "timestamp": "${TIMESTAMP}",
  "config": {
    "mpi_ranks": ${NP},
    "problem_args": "${MG_ARGS}",
    "baseline_runs": ${BASELINE_RUNS}
  },
  "gprof_hotspots": ${gprof_hotspots},
  "perf_hotspots": ${perf_hotspots},
  "report_files": {
    "baseline_summary": "baseline_summary_${TIMESTAMP}.txt",
    "gprof_full": "gprof_full_${TIMESTAMP}.txt",
    "gprof_top": "gprof_top${TOP_N}_${TIMESTAMP}.txt",
    "perf_stat": "perf_stat_${TIMESTAMP}.txt",
    "perf_report": "perf_report_${TIMESTAMP}.txt"
  },
  "next_step": "将 gprof_hotspots / perf_hotspots 中的函数名提取，在源码中定位对应代码，通过 LLM API 分析其 GPU 加速可行性"
}
ENDJSON

  log "汇总报告 ✓"
  echo ""
  echo "=============================================="
  echo "        Profiling 完成！报告目录："
  echo "        $REPORT_DIR"
  echo "=============================================="
  echo ""
  echo "生成的文件："
  ls -lh "$REPORT_DIR/"*"${TIMESTAMP}"* 2>/dev/null
  echo ""
  echo "JSON 汇总（供 LLM 消费）："
  cat "$summary_file"
  echo ""
  echo ""
  echo "=== 下一步建议 ==="
  echo "1. 查看 JSON 中的热点函数列表"
  echo "2. 在 miniGhost 源码中定位这些函数："
  echo "     grep -rn 'FUNCTION_NAME' $SRC_DIR/"
  echo "3. 将函数源码喂给 LLM，分析 GPU 加速可行性"
  echo "4. 如需更精细分析，在 HPC 上启用 VTune："
  echo "     ENABLE_VTUNE=1 ./profile_miniGhost.sh"
}

# ======================== 主流程 ========================
main() {
  log "=========================================="
  log " miniGhost Profiling Pipeline"
  log " MPI ranks: $NP | Baseline runs: $BASELINE_RUNS"
  log "=========================================="

  phase0_build
  phase1_baseline
  phase2_gprof
  phase3_perf
  phase4_vtune
  phase5_summary
}

main "$@"
