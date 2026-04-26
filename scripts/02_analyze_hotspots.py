#!/usr/bin/env python3
"""
02_analyze_hotspots.py — Universal hotspot analysis: extract source code → LLM → GPU feasibility report.

Reads hotspot keywords and source configuration from app_config.json.
Supports Fortran, C, and C++ source files.

Usage:
    source .env
    python3 scripts/02_analyze_hotspots.py miniGhost
    python3 scripts/02_analyze_hotspots.py udunits
"""

import os
import sys
import json
import glob
import re
from datetime import datetime
from pathlib import Path

# ============================================================
# Import shared config loader
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config_loader import load_app_config, get_app_paths
except ImportError:
    # Fallback: inline config loading
    def load_app_config(app_name):
        config_path = PROJECT_ROOT / "app_config.json"
        with open(config_path) as f:
            data = json.load(f)
        if app_name not in data:
            print(f"[ERROR] Unknown app: {app_name}")
            sys.exit(1)
        return data[app_name]

    def get_app_paths(app_name):
        config = load_app_config(app_name)
        return {
            "project_root": PROJECT_ROOT,
            "src_dir": PROJECT_ROOT / "apps" / app_name / config.get("src_subdir", "src"),
            "report_dir": PROJECT_ROOT / "reports" / app_name,
        }


# ============================================================
# Configuration
# ============================================================
LLM_PROVIDER = "auto"  # "openai" | "anthropic" | "auto"
OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
MAX_LINES_PER_FUNCTION = 200


# ============================================================
# Step 1: Locate hotspot function source code (language-aware)
# ============================================================
# File extensions per language
LANG_EXTENSIONS = {
    "fortran": ["*.F", "*.f", "*.F90", "*.f90", "*.f95"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.hpp", "*.h", "*.cc", "*.cxx"],
}

# Function detection patterns per language
FUNC_PATTERNS = {
    "fortran": {
        "start": r"\b(subroutine|function)\s+\w*{keyword}\w*",
        "end": r"^\s*end\s+(subroutine|function)",
        "exclude_start": r"^\s*end\s+",
        "name_extract": r"(subroutine|function)\s+(\w+)",
    },
    "c": {
        # Match C function definitions: return_type func_name(...)  {
        "start": r"^[a-zA-Z_][\w\s\*]*\b{keyword}\w*\s*\(",
        "end": r"^}}",  # closing brace at column 0
        "exclude_start": r"^\s*(//|/\*|\*|#)",  # skip comments and preprocessor
        "name_extract": r"\b(\w+{keyword}\w*)\s*\(",
    },
    "cpp": {
        "start": r"^[a-zA-Z_][\w\s\*\&:]*\b{keyword}\w*\s*\(",
        "end": r"^}}",
        "exclude_start": r"^\s*(//|/\*|\*|#)",
        "name_extract": r"\b(\w+{keyword}\w*)\s*\(",
    },
}


def find_function_in_source(src_dir: str, keyword: str, language: str) -> list:
    """Search for functions matching a keyword in source files."""
    results = []
    extensions = LANG_EXTENSIONS.get(language, ["*.c", "*.h", "*.cpp", "*.hpp"])
    patterns = FUNC_PATTERNS.get(language, FUNC_PATTERNS["c"])

    start_pattern = patterns["start"].replace("{keyword}", keyword)
    end_pattern = patterns["end"]
    exclude_pattern = patterns.get("exclude_start", "")
    name_pattern = patterns["name_extract"].replace("{keyword}", keyword)

    for ext in extensions:
        for fpath in glob.glob(os.path.join(src_dir, "**", ext), recursive=True):
            with open(fpath, "r", errors="replace") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                # Check if this line starts a function matching our keyword
                if not re.search(start_pattern, line, re.IGNORECASE):
                    continue

                # Exclude lines that are just END statements (Fortran)
                if exclude_pattern and re.search(exclude_pattern, line, re.IGNORECASE):
                    continue

                # Find the end of the function
                start = i
                end = min(i + MAX_LINES_PER_FUNCTION, len(lines))

                if language == "fortran":
                    for j in range(i + 1, min(i + MAX_LINES_PER_FUNCTION, len(lines))):
                        if re.search(end_pattern, lines[j], re.IGNORECASE):
                            end = j + 1
                            break
                else:
                    # For C/C++: count braces
                    brace_count = 0
                    found_open = False
                    for j in range(i, min(i + MAX_LINES_PER_FUNCTION, len(lines))):
                        brace_count += lines[j].count("{") - lines[j].count("}")
                        if "{" in lines[j]:
                            found_open = True
                        if found_open and brace_count <= 0:
                            end = j + 1
                            break

                func_code = "".join(lines[start:end])

                # Extract function name
                func_name = keyword
                name_match = re.search(name_pattern, line, re.IGNORECASE)
                if name_match:
                    func_name = name_match.group(2) if language == "fortran" else name_match.group(1)

                results.append({
                    "function_name": func_name,
                    "file": os.path.relpath(fpath, src_dir),
                    "start_line": start + 1,
                    "end_line": end,
                    "source_code": func_code,
                })
    return results


# ============================================================
# Step 2: Load profiling data
# ============================================================
def load_profiling_data(report_dir: str) -> dict:
    json_files = sorted(glob.glob(os.path.join(report_dir, "profiling_summary_*.json")))
    if not json_files:
        print(f"[WARN] No profiling JSON found in {report_dir}")
        return {}
    latest = json_files[-1]
    print(f"[INFO] Reading: {latest}")
    with open(latest) as f:
        return json.load(f)


# ============================================================
# Step 3: Construct the Prompt (language-aware)
# ============================================================
def build_prompt(profiling_data: dict, functions: list, app_config: dict) -> str:
    language = app_config.get("language", "unknown")
    domain = app_config.get("domain", "scientific computing")
    gpu_strategy = app_config.get("gpu_patch", {}).get("strategy", "CUDA/OpenACC")

    lang_display = {"fortran": "Fortran", "c": "C", "cpp": "C++"}.get(language, language)

    prompt = f"""You are an HPC performance optimization expert, proficient in {lang_display}, MPI, CUDA, OpenACC, and GPU programming.

## Task
Analyze the following performance hotspot functions from a {domain} application written in {lang_display}.
Evaluate the feasibility of accelerating them on the GPU. The intended porting target for this application is {gpu_strategy}, but you should also consider and score alternative programming models so that the final choice can be read against the alternatives.

## Profiling Data Overview
"""
    if profiling_data:
        prompt += f"- Application: {profiling_data.get('application', 'N/A')}\n"
        config = profiling_data.get("config", {})
        prompt += f"- MPI ranks: {config.get('mpi_ranks', 'N/A')}\n"
        prompt += f"- Problem Size: {config.get('problem_args', 'N/A')}\n"
        prompt += f"- Build system: {config.get('build_system', 'N/A')}\n\n"
        prompt += "### gprof Hotspot Ranking\n"
        prompt += "| Rank | Function | % Time | Self (s) | Calls |\n"
        prompt += "|------|----------|--------|----------|-------|\n"
        for i, h in enumerate(profiling_data.get("gprof_hotspots", [])):
            if h.get("function") == "name":
                continue
            prompt += f"| {i} | `{h['function']}` | {h['pct_time']}% | {h['self_sec']}s | {h['calls']} |\n"
        prompt += "\n"

    prompt += "## Hotspot Function Source Code\n\n"
    for func in functions:
        prompt += f"### {func['function_name']} ({func['file']}:{func['start_line']}-{func['end_line']})\n"
        prompt += f"```{lang_display.lower()}\n{func['source_code']}```\n\n"

    prompt += f"""## For each hotspot function, please answer the following:

### Hotspot-level analysis (the same five dimensions for every hotspot)

1. **Function Summary**: A one-sentence description.
2. **Computational Characteristics**: Compute-bound / Memory-bound / Communication-bound?
3. **Data Dependency Analysis**: Are there loop-carried dependencies? Is it parallelizable?
4. **Implementation Difficulty**: Low / Medium / High.
5. **Pseudocode / Conceptual Approach**: A kernel sketch or pseudocode in {lang_display}.

### Candidate porting strategies (propose 2-3, each scored independently)

For each candidate strategy, fill in the following three dimensions. The intended target for this application is `{gpu_strategy}`, which should appear as one of the candidates; the others should be plausible alternatives (for example, CUDA, OpenACC, OpenMP target offload, or a hybrid approach).

#### Candidate A: <short name>
- **Programming Model**: e.g., CUDA / OpenACC / OpenMP target / Kokkos.
- **GPU Acceleration Feasibility Score (1-10)**: how well this candidate maps to the GPU for this hotspot.
- **Expected Speedup**: A rough estimate (e.g., 5x, 20x).
- **One-sentence rationale**: why this score and speedup.

#### Candidate B: <short name>
(repeat the four bullets above)

#### Candidate C: <short name>  *(optional, only if a third candidate is materially different)*
(repeat the four bullets above)

### After all hotspots have been analysed

Provide an **optimisation priority ranking** across hotspots, identifying which one is most worth porting first. Please format your entire response in Markdown.
"""
    return prompt


# ============================================================
# Step 4: Call LLM (unchanged)
# ============================================================
def call_openai(prompt: str) -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    print(f"[INFO] Calling OpenAI ({OPENAI_MODEL})...")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a performance optimization expert proficient in HPC and GPU programming."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def call_anthropic(prompt: str) -> str:
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Please set the ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    print(f"[INFO] Calling Anthropic ({ANTHROPIC_MODEL})...")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        system="You are a performance optimization expert proficient in HPC and GPU programming.",
    )
    return response.content[0].text


def call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "openai" or (
        LLM_PROVIDER == "auto" and os.environ.get("OPENAI_API_KEY")
    ):
        return call_openai(prompt)
    elif LLM_PROVIDER == "anthropic" or (
        LLM_PROVIDER == "auto" and os.environ.get("ANTHROPIC_API_KEY")
    ):
        return call_anthropic(prompt)
    else:
        print("Error: No API key detected. Please:")
        print("  export OPENAI_API_KEY='sk-xxx'  OR")
        print("  export ANTHROPIC_API_KEY='sk-ant-xxx'")
        sys.exit(1)


# ============================================================
# Step 5: Save Report
# ============================================================
def save_report(prompt: str, response: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    prompt_file = os.path.join(output_dir, f"llm_prompt_{timestamp}.md")
    with open(prompt_file, "w") as f:
        f.write(prompt)

    report_file = os.path.join(output_dir, f"llm_analysis_{timestamp}.md")
    with open(report_file, "w") as f:
        f.write(f"# GPU Acceleration Feasibility Analysis Report\n\n")
        f.write(f"_Generated at: {datetime.now().isoformat()}_\n\n---\n\n")
        f.write(response)

    print(f"[INFO] Prompt saved to:  {prompt_file}")
    print(f"[INFO] Report saved to:  {report_file}")
    return report_file


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/02_analyze_hotspots.py <app_name>")
        print("Example: python3 scripts/02_analyze_hotspots.py miniGhost")
        sys.exit(1)

    app_name = sys.argv[1]
    app_config = load_app_config(app_name)
    paths = get_app_paths(app_name)

    src_dir = str(paths["src_dir"])
    report_dir = str(paths["report_dir"])
    language = app_config["language"]

    print("=" * 60)
    print(f"  {app_name} ({language}) Hotspot → LLM GPU Analysis")
    print("=" * 60)

    # Step 1: Search for hotspot functions
    keywords = app_config.get("profiling", {}).get("hotspot_keywords", [])
    if not keywords:
        print(f"[WARN] No hotspot keywords configured for {app_name}.")
        print(f"  Add them to app_config.json → {app_name} → profiling → hotspot_keywords")
        sys.exit(1)

    print(f"\n[Step 1] Searching for hotspot functions in {src_dir}...")
    all_functions = []
    for kw in keywords:
        found = find_function_in_source(src_dir, kw, language)
        for f in found:
            print(f"  ✓ {f['function_name']} found in {f['file']}:{f['start_line']}-{f['end_line']}")
            all_functions.append(f)
        if not found:
            print(f"  ✗ Keyword '{kw}' not found.")

    if not all_functions:
        print("[ERROR] No hotspot functions found.")
        sys.exit(1)

    # Step 2: Load profiling data
    print(f"\n[Step 2] Loading profiling report...")
    profiling_data = load_profiling_data(report_dir)

    # Step 3: Construct prompt
    print(f"\n[Step 3] Constructing prompt...")
    prompt = build_prompt(profiling_data, all_functions, app_config)
    print(f"  Length: {len(prompt)} characters, ~{len(prompt)//4} tokens")

    # Step 4: Call LLM
    print(f"\n[Step 4] Calling LLM...")
    response = call_llm(prompt)

    # Step 5: Save
    print(f"\n[Step 5] Saving report...")
    report_file = save_report(prompt, response, report_dir)

    print(f"\n{'='*60}")
    print(f"  Complete! Report saved to: {report_file}")
    print(f"{'='*60}")
    print(f"\n{response[:2000]}")
    if len(response) > 2000:
        print(f"\n... (Total {len(response)} characters, see file for full report)")


if __name__ == "__main__":
    main()