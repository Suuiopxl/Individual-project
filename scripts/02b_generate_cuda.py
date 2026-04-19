#!/usr/bin/env python3
"""
02b_generate_cuda.py — Interactive Phase B code-generation session.

Continues from the most recent Phase A feasibility report produced by
02_analyze_hotspots.py, and launches a multi-turn REPL conversation with
the language model. Each turn is saved to a timestamped transcript under
reports/<app>/ so that nothing is lost if the session is interrupted.
Full conversation history is kept in memory and replayed on every API
call, which is what gives the agent continuity across turns.

Usage:
    source .env
    python3 scripts/02b_generate_cuda.py <app_name>

Session commands:
    /quit              End the session and save the transcript.
    /save              Force-save the transcript now.
    /history           Print a short summary of every turn so far.
    /reset             Clear the conversation back to the Phase A context.
    \\ at end of line  Continue input onto the next line.
"""

import os
import sys
import glob
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================
# Project config loading (same pattern as 02_analyze_hotspots.py)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config_loader import load_app_config, get_app_paths
except ImportError:
    import json
    def load_app_config(app_name):
        with open(PROJECT_ROOT / "app_config.json") as f:
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
# LLM configuration
# ============================================================
LLM_PROVIDER = "auto"   # "openai" | "anthropic" | "auto"
OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS_PER_REPLY = 4096

# Warning threshold on cumulative transcript size (rough proxy for context usage)
CONTEXT_WARN_CHARS = 400_000

SYSTEM_PROMPT = """\
You are a GPU code generation assistant for high-performance computing (HPC)
scientific applications. Your first message in this conversation is a structured
feasibility analysis that you produced for the user's application in an earlier
phase. The user will now ask you to generate and iteratively refine GPU code
(typically CUDA or OpenACC) for the hotspots discussed in that analysis.

Guidelines:
- Stay grounded in the specific source code and performance characteristics from the analysis.
- When producing code, use fenced code blocks with a language tag (```cpp, ```c, ```fortran, or ```cuda).
- Briefly justify design decisions: memory layout, kernel shape, launch configuration.
- When asked to revise, preserve what the user has accepted and change only what is necessary.
- If the user reports a compile error, quote the relevant line of code before proposing a fix.
"""


# ============================================================
# Phase A context loading
# ============================================================
def load_phase_a_context(app_name: str, paths: dict) -> tuple:
    """Find the most recent llm_analysis_*.md produced by 02_analyze_hotspots.py."""
    report_dir = paths["report_dir"]
    files = sorted(glob.glob(str(report_dir / "llm_analysis_*.md")))
    if not files:
        print(f"[ERROR] No Phase A analysis found for {app_name} in {report_dir}")
        print(f"        Run first: python3 scripts/02_analyze_hotspots.py {app_name}")
        sys.exit(1)
    latest = files[-1]
    with open(latest) as f:
        content = f.read()
    return latest, content


# ============================================================
# Input handling: single-line or \\-continued multi-line
# ============================================================
def read_user_message(prompt: str = "You> ") -> str:
    """Read a user turn. Supports \\ line-continuation for multi-line input."""
    lines = []
    cur_prompt = prompt
    while True:
        try:
            line = input(cur_prompt)
        except (EOFError, KeyboardInterrupt):
            print()  # clean line after Ctrl-C / Ctrl-D
            return "/quit"
        if line.endswith("\\"):
            lines.append(line[:-1])
            cur_prompt = "...> "
        else:
            lines.append(line)
            break
    return "\n".join(lines).strip()


# ============================================================
# LLM provider wrappers (messages-history-aware)
# ============================================================
def call_openai(messages: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=full_messages,
        max_tokens=MAX_TOKENS_PER_REPLY,
    )
    return response.choices[0].message.content or ""


def call_anthropic(messages: list) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS_PER_REPLY,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


def call_llm(messages: list) -> str:
    """Dispatch to OpenAI or Anthropic depending on available credentials."""
    if LLM_PROVIDER == "openai" or (
        LLM_PROVIDER == "auto" and os.environ.get("OPENAI_API_KEY")
    ):
        return call_openai(messages)
    elif LLM_PROVIDER == "anthropic" or (
        LLM_PROVIDER == "auto" and os.environ.get("ANTHROPIC_API_KEY")
    ):
        return call_anthropic(messages)
    else:
        print("Error: No API key detected.")
        print("  export OPENAI_API_KEY='sk-...'  OR")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)


# ============================================================
# Transcript IO
# ============================================================
def save_transcript(messages: list, transcript_path: Path,
                    phase_a_source: str, app_name: str) -> None:
    """Write the full conversation to disk as a markdown file."""
    with open(transcript_path, "w") as f:
        f.write(f"# {app_name} --- Phase B Code Generation Session\n\n")
        f.write(f"_Started: {datetime.now().isoformat()}_  \n")
        f.write(f"_Phase A source: `{os.path.basename(phase_a_source)}`_  \n")
        f.write(f"_Provider: {LLM_PROVIDER} / model: {OPENAI_MODEL}_\n\n")
        f.write("---\n\n")
        for i, msg in enumerate(messages):
            role = msg["role"].capitalize()
            f.write(f"## Turn {i}: {role}\n\n{msg['content']}\n\n---\n\n")


def total_chars(messages: list) -> int:
    return sum(len(m["content"]) for m in messages)


# ============================================================
# Main REPL loop
# ============================================================
def seed_messages(app_name: str, phase_a_content: str) -> list:
    """Seed the conversation with the Phase A report as the first assistant turn."""
    return [
        {
            "role": "user",
            "content": f"Please analyse the GPU-acceleration feasibility of the hotspots in my {app_name} application.",
        },
        {
            "role": "assistant",
            "content": phase_a_content,
        },
    ]


def print_banner(app_name: str, transcript_path: Path) -> None:
    print("=" * 64)
    print(f"  {app_name} --- Interactive Phase B session")
    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Transcript: {transcript_path.name}")
    print(f"  Commands: /quit  /save  /history  /reset")
    print(f"  Multi-line input: end a line with \\")
    print("=" * 64)


def print_response(response: str) -> None:
    print("\n" + "-" * 64)
    print(response)
    print("-" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("app_name", help="Application name as declared in app_config.json")
    args = parser.parse_args()

    app_name = args.app_name
    app_config = load_app_config(app_name)
    paths = get_app_paths(app_name)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    # Locate the Phase A report to seed context
    phase_a_path, phase_a_content = load_phase_a_context(app_name, paths)
    print(f"[INFO] Loaded Phase A analysis: {os.path.basename(phase_a_path)}")

    # Initialise transcript path for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = paths["report_dir"] / f"session_phase_b_{timestamp}.md"

    # Seed conversation
    messages = seed_messages(app_name, phase_a_content)

    print_banner(app_name, transcript_path)

    while True:
        user_input = read_user_message()
        if not user_input:
            continue

        # Session commands
        if user_input == "/quit":
            save_transcript(messages, transcript_path, phase_a_path, app_name)
            print(f"\n[INFO] Transcript saved: {transcript_path}")
            break
        elif user_input == "/save":
            save_transcript(messages, transcript_path, phase_a_path, app_name)
            print(f"[INFO] Transcript saved: {transcript_path}")
            continue
        elif user_input == "/history":
            print("\n=== Conversation history ===")
            for i, msg in enumerate(messages):
                preview = msg["content"].strip().replace("\n", " ")[:120]
                print(f"  [{i}] {msg['role']:10s} {preview}{'...' if len(msg['content']) > 120 else ''}")
            print(f"  Total: {total_chars(messages):,} chars across {len(messages)} turns\n")
            continue
        elif user_input == "/reset":
            messages = seed_messages(app_name, phase_a_content)
            print("[INFO] Conversation reset to Phase A context.")
            continue

        # Normal turn: append user message, query LLM, append response, save
        messages.append({"role": "user", "content": user_input})

        try:
            print("[INFO] Querying LLM...")
            response = call_llm(messages)
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            messages.pop()  # remove the user message whose reply we failed to get
            continue

        messages.append({"role": "assistant", "content": response})
        save_transcript(messages, transcript_path, phase_a_path, app_name)
        print_response(response)

        # Soft warning on context growth
        if total_chars(messages) > CONTEXT_WARN_CHARS:
            print(f"[WARN] Conversation is {total_chars(messages):,} chars; consider /reset soon.\n")


if __name__ == "__main__":
    main()