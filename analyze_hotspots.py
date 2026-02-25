#!/usr/bin/env python3
"""
analyze_hotspots.py — 将 miniGhost profiling 热点函数源码喂给 LLM 分析 GPU 加速可行性

使用方法：
  1. 安装依赖：
       pip install openai anthropic

  2. 设置 API key（二选一）：
       export OPENAI_API_KEY="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"        # OpenAI
       export ANTHROPIC_API_KEY="sk-ant-xxxxx"  # Anthropic Claude

  3. 运行：
       python3 analyze_hotspots.py

  4. 输出会保存到 analysis_report_<timestamp>.md
"""

import os
import sys
import json
import glob
import subprocess
import re
from datetime import datetime
from pathlib import Path

# ============================================================
# 配置区 —— 根据你的环境修改
# ============================================================
MINIGHOST_SRC_DIR = os.path.expanduser("~/miniGhost/ref")
PROFILING_REPORT_DIR = os.path.expanduser("~/miniGhost_profiling/reports")

# LLM 配置（二选一，优先使用设置了 API key 的那个）
LLM_PROVIDER = "auto"  # "openai" | "anthropic" | "auto"
OPENAI_MODEL = "gpt-5-mini"  # 便宜且够用，$0.25/1M input tokens
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# 从 gprof 提取的热点函数关键词（只保留 Top 3 核心热点）
HOTSPOT_KEYWORDS = [
    "mg_stencil_2d5pt",
    "mg_sum_grid",
    "mg_flux_accumulate_2d5pt_3d7pt",
]

# 每个函数提取的最大行数（避免 token 过多）
MAX_LINES_PER_FUNCTION = 200


# ============================================================
# Step 1: 在源码目录中定位热点函数
# ============================================================
def find_function_in_source(src_dir: str, keyword: str) -> list[dict]:
    """
    用 grep 在源码中搜索包含 keyword 的 subroutine/function 定义，
    然后提取整个函数体。
    """
    results = []

    # 搜索所有 Fortran 文件
    for ext in ["*.F", "*.f", "*.F90", "*.f90", "*.f95"]:
        for fpath in glob.glob(os.path.join(src_dir, "**", ext), recursive=True):
            with open(fpath, "r", errors="replace") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                # 匹配 subroutine 或 function 定义行（排除 END SUBROUTINE）
                if re.search(
                    rf"\b(subroutine|function)\s+\w*{keyword}\w*",
                    line,
                    re.IGNORECASE,
                ) and not re.search(r"^\s*end\s+", line, re.IGNORECASE):
                    # 找到函数起始，现在提取到 END SUBROUTINE/FUNCTION
                    start = i
                    end = min(i + MAX_LINES_PER_FUNCTION, len(lines))

                    for j in range(i + 1, min(i + MAX_LINES_PER_FUNCTION, len(lines))):
                        if re.search(
                            r"^\s*end\s+(subroutine|function)",
                            lines[j],
                            re.IGNORECASE,
                        ):
                            end = j + 1
                            break

                    func_code = "".join(lines[start:end])
                    func_name_match = re.search(
                        r"(subroutine|function)\s+(\w+)", line, re.IGNORECASE
                    )
                    func_name = (
                        func_name_match.group(2) if func_name_match else keyword
                    )

                    results.append(
                        {
                            "function_name": func_name,
                            "file": os.path.relpath(fpath, src_dir),
                            "start_line": start + 1,
                            "end_line": end,
                            "source_code": func_code,
                        }
                    )

    return results


# ============================================================
# Step 2: 读取 profiling 报告
# ============================================================
def load_profiling_data(report_dir: str) -> dict:
    """读取最新的 JSON 汇总报告"""

    json_files = sorted(glob.glob(os.path.join(report_dir, "profiling_summary_*.json")))
    if not json_files:
        print(f"[WARN] 未找到 profiling JSON 报告，在 {report_dir}")
        return {}

    latest = json_files[-1]
    print(f"[INFO] 读取 profiling 报告: {latest}")
    with open(latest) as f:
        return json.load(f)


# ============================================================
# Step 3: 构造 LLM Prompt
# ============================================================
def build_prompt(profiling_data: dict, functions: list[dict]) -> str:
    """
    构造给 LLM 的 prompt，包含：
    - profiling 概览
    - 每个热点函数的源码
    - 分析要求
    """

    prompt = """你是一位 HPC 性能优化专家，精通 Fortran、MPI、CUDA、OpenACC 和 GPU 编程。

## 任务
分析以下 miniGhost（一个 3D 热传导差分模拟 mini-app）的性能热点函数，评估其 GPU 加速的可行性。

## Profiling 数据概览
"""

    if profiling_data:
        prompt += f"- 应用: {profiling_data.get('application', 'miniGhost')}\n"
        config = profiling_data.get("config", {})
        prompt += f"- MPI ranks: {config.get('mpi_ranks', 'N/A')}\n"
        prompt += f"- 问题规模: {config.get('problem_args', 'N/A')}\n\n"

        prompt += "### gprof 热点函数排名\n"
        prompt += "| 排名 | 函数 | 时间占比 | Self (s) | 调用次数 |\n"
        prompt += "|------|------|----------|----------|----------|\n"
        for i, h in enumerate(profiling_data.get("gprof_hotspots", [])):
            if h.get("function") == "name":  # 跳过表头
                continue
            prompt += f"| {i} | `{h['function']}` | {h['pct_time']}% | {h['self_sec']}s | {h['calls']} |\n"
        prompt += "\n"

    prompt += "## 热点函数源码\n\n"

    for func in functions:
        prompt += f"### {func['function_name']} ({func['file']}:{func['start_line']}-{func['end_line']})\n"
        prompt += f"```fortran\n{func['source_code']}```\n\n"

    prompt += """## 请对每个函数回答以下问题

对每个热点函数，请提供：

1. **功能摘要**: 这个函数在做什么（一句话）
2. **计算特征**: 计算密集型 / 内存密集型 / 通信密集型？
3. **数据依赖分析**: 迭代间是否有数据依赖？是否可并行？
4. **GPU 加速可行性评分**: 1-10 分（10=非常适合 GPU）
5. **推荐的 GPU 化方案**: CUDA kernel / OpenACC / Kokkos / 不建议 GPU 化
6. **预期加速比**: 粗略估算
7. **实现难度**: 低 / 中 / 高
8. **具体建议**: 如何改写为 GPU 代码（伪代码或关键思路）

最后，请给出一个**优化优先级排序**，说明应该先 GPU 化哪个函数，为什么。

请用 Markdown 格式回答。
"""

    return prompt


# ============================================================
# Step 4: 调用 LLM API
# ============================================================
def call_openai(prompt: str) -> str:
    """调用 OpenAI API (GPT-4o)"""
    try:
        from openai import OpenAI
    except ImportError:
        print("请先安装: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        print("  export OPENAI_API_KEY='sk-xxxxxx'")
        sys.exit(1)

    print(f"[INFO] 调用 OpenAI API ({OPENAI_MODEL})...")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一位精通 HPC、Fortran 和 GPU 编程的性能优化专家。",
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=32768,
    )

    # 调试：打印完整响应结构
    print(f"[DEBUG] Response object: {response}")
    print(f"[DEBUG] Choices count: {len(response.choices)}")
    if response.choices:
        choice = response.choices[0]
        print(f"[DEBUG] Finish reason: {choice.finish_reason}")
        print(f"[DEBUG] Message role: {choice.message.role}")
        print(f"[DEBUG] Message content type: {type(choice.message.content)}")
        print(f"[DEBUG] Message content repr: {repr(choice.message.content)[:500]}")
        # 某些 reasoning 模型把输出放在 content 外面
        if hasattr(choice.message, 'reasoning_content'):
            print(f"[DEBUG] Has reasoning_content: {repr(choice.message.reasoning_content)[:500]}")

    result = response.choices[0].message.content or ""

    # 如果 content 为空，尝试其他字段
    if not result.strip():
        print("[WARN] content 为空，尝试读取其他字段...")
        msg = response.choices[0].message
        # 某些模型把内容放在 refusal 或其他属性
        for attr in ['reasoning_content', 'refusal', 'tool_calls']:
            if hasattr(msg, attr) and getattr(msg, attr):
                print(f"[DEBUG] Found content in '{attr}': {repr(getattr(msg, attr))[:300]}")

    if not result.strip():
        print("[WARN] API 返回空内容。尝试换用 gpt-4.1-mini 模型...")
        response2 = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位精通 HPC、Fortran 和 GPU 编程的性能优化专家。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        result = response2.choices[0].message.content or ""

    return result


def call_anthropic(prompt: str) -> str:
    """调用 Anthropic Claude API"""
    try:
        import anthropic
    except ImportError:
        print("请先安装: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 ANTHROPIC_API_KEY")
        print("  export ANTHROPIC_API_KEY='sk-ant-xxxxxx'")
        sys.exit(1)

    print(f"[INFO] 调用 Anthropic API ({ANTHROPIC_MODEL})...")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        system="你是一位精通 HPC、Fortran 和 GPU 编程的性能优化专家。",
    )

    return response.content[0].text


def call_llm(prompt: str) -> str:
    """根据配置和可用的 API key 自动选择 LLM provider"""

    if LLM_PROVIDER == "openai" or (
        LLM_PROVIDER == "auto" and os.environ.get("OPENAI_API_KEY")
    ):
        return call_openai(prompt)
    elif LLM_PROVIDER == "anthropic" or (
        LLM_PROVIDER == "auto" and os.environ.get("ANTHROPIC_API_KEY")
    ):
        return call_anthropic(prompt)
    else:
        print("错误: 未检测到任何 API key。请设置以下之一:")
        print("  export OPENAI_API_KEY='sk-xxxxxx'")
        print("  export ANTHROPIC_API_KEY='sk-ant-xxxxxx'")
        sys.exit(1)


# ============================================================
# Step 5: 保存分析报告
# ============================================================
def save_report(prompt: str, response: str, output_dir: str) -> str:
    """保存 prompt 和 LLM 响应到文件"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整 prompt（方便复现）
    prompt_file = os.path.join(output_dir, f"llm_prompt_{timestamp}.md")
    with open(prompt_file, "w") as f:
        f.write(prompt)
    print(f"[INFO] Prompt 已保存: {prompt_file}")

    # 保存 LLM 分析报告
    report_file = os.path.join(output_dir, f"analysis_report_{timestamp}.md")
    with open(report_file, "w") as f:
        f.write(f"# miniGhost GPU 加速可行性分析报告\n\n")
        f.write(f"_生成时间: {datetime.now().isoformat()}_\n\n")
        f.write(f"---\n\n")
        f.write(response)
    print(f"[INFO] 分析报告已保存: {report_file}")

    return report_file


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  miniGhost 热点函数 → LLM GPU 加速分析")
    print("=" * 60)

    # Step 1: 定位源码中的热点函数
    print(f"\n[Step 1] 在 {MINIGHOST_SRC_DIR} 中搜索热点函数...")
    all_functions = []
    for keyword in HOTSPOT_KEYWORDS:
        found = find_function_in_source(MINIGHOST_SRC_DIR, keyword)
        if found:
            for f in found:
                print(f"  ✓ 找到 {f['function_name']} in {f['file']}:{f['start_line']}-{f['end_line']}")
                all_functions.append(f)
        else:
            print(f"  ✗ 未找到包含 '{keyword}' 的函数")

    if not all_functions:
        print("\n[ERROR] 未找到任何热点函数源码！")
        print("请检查 MINIGHOST_SRC_DIR 路径是否正确")
        print(f"当前路径: {MINIGHOST_SRC_DIR}")
        print("目录内容:")
        if os.path.isdir(MINIGHOST_SRC_DIR):
            for f in os.listdir(MINIGHOST_SRC_DIR):
                print(f"  {f}")
        sys.exit(1)

    # Step 2: 读取 profiling 数据
    print(f"\n[Step 2] 读取 profiling 报告...")
    profiling_data = load_profiling_data(PROFILING_REPORT_DIR)

    # Step 3: 构造 prompt
    print(f"\n[Step 3] 构造 LLM prompt...")
    prompt = build_prompt(profiling_data, all_functions)
    print(f"  Prompt 长度: {len(prompt)} 字符, 约 {len(prompt)//4} tokens")

    # Step 4: 调用 LLM
    print(f"\n[Step 4] 调用 LLM API...")
    response = call_llm(prompt)

    # Step 5: 保存报告
    print(f"\n[Step 5] 保存分析报告...")
    report_file = save_report(prompt, response, PROFILING_REPORT_DIR)

    print(f"\n{'=' * 60}")
    print(f"  完成！分析报告: {report_file}")
    print(f"{'=' * 60}")

    # 打印摘要
    print(f"\n--- LLM 分析摘要 (前 2000 字) ---\n")
    print(response[:2000])
    if len(response) > 2000:
        print(f"\n... (完整报告共 {len(response)} 字，请查看文件)")


if __name__ == "__main__":
    main()
