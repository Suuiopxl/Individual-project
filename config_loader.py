"""
config_loader.py — Shared configuration loader for all pipeline scripts.

All scripts import this module to get app-specific settings from app_config.json.

Usage:
    from config_loader import load_app_config, get_project_root

    config = load_app_config("miniGhost")
    print(config["build"]["makefile"])  # "makefile.mpi.gnu"
"""

import json
import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Locate the project root directory.
    Works whether called from scripts/ or from project root.
    """
    # Try: this file lives in scripts/
    candidate = Path(__file__).resolve().parent.parent
    if (candidate / "app_config.json").exists():
        return candidate

    # Try: this file lives in project root
    candidate = Path(__file__).resolve().parent
    if (candidate / "app_config.json").exists():
        return candidate

    # Fallback: search upward
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "app_config.json").exists():
            return current
        current = current.parent

    print("[ERROR] Cannot locate project root (no app_config.json found).")
    sys.exit(1)


def load_all_configs() -> dict:
    """Load the entire app_config.json."""
    project_root = get_project_root()
    config_path = project_root / "app_config.json"

    with open(config_path, "r") as f:
        data = json.load(f)

    # Remove comment keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_app_config(app_name: str) -> dict:
    """Load configuration for a specific application."""
    all_configs = load_all_configs()

    if app_name not in all_configs:
        available = ", ".join(all_configs.keys())
        print(f"[ERROR] Unknown application: '{app_name}'")
        print(f"  Available applications: {available}")
        print(f"  To add a new app, edit app_config.json")
        sys.exit(1)

    return all_configs[app_name]


def list_apps() -> list:
    """Return a list of all configured application names."""
    return list(load_all_configs().keys())


def get_app_paths(app_name: str) -> dict:
    """
    Return standardized paths for an application.
    Enforces the unified directory convention.
    """
    root = get_project_root()
    config = load_app_config(app_name)

    return {
        "project_root": root,
        "app_dir": root / "apps" / app_name,
        "src_dir": root / "apps" / app_name / config.get("src_subdir", "src"),
        "gpu_dir": root / "apps" / f"{app_name}_gpu",
        "gpu_src_dir": root / "apps" / f"{app_name}_gpu" / config.get("src_subdir", "src"),
        "backup_dir": root / "apps" / f"{app_name}_gpu" / "original_backup",
        "build_dir": root / "apps" / f"{app_name}_build",
        "report_dir": root / "reports" / app_name,
        "comparison_dir": root / "reports" / app_name / "performance_comparison",
    }


if __name__ == "__main__":
    # Quick test: show all configured apps
    print("Configured applications:")
    for name in list_apps():
        config = load_app_config(name)
        print(f"  {name}: {config['language']} — {config['domain']}")