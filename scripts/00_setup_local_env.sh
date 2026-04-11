#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 00_setup_local_env.sh — Universal environment setup
#
# Reads app_config.json to know which repos to clone and what dependencies
# to install. Supports adding new apps without modifying this script.
#
# Usage:
#   chmod +x scripts/00_setup_local_env.sh
#   ./scripts/00_setup_local_env.sh              # Set up all apps
#   ./scripts/00_setup_local_env.sh miniGhost     # Set up specific app only
###############################################################################

log()  { echo -e "\n\033[1;32m[$(date +%H:%M:%S)] $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
err()  { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; }

# ======================== Project root ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If running standalone (not from scripts/), assume current dir
if [ ! -f "$PROJECT_ROOT/app_config.json" ]; then
    PROJECT_ROOT="$(pwd)"
fi

cd "$PROJECT_ROOT"
CONFIG_FILE="$PROJECT_ROOT/app_config.json"
TARGET_APP="${1:-all}"

log "=========================================="
log " Environment Setup"
log " Project: $PROJECT_ROOT"
log " Target:  $TARGET_APP"
log "=========================================="

# ======================== 1. System detection ========================
log "1. Detecting system environment"

if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  System: WSL2"
else
    echo "  System: Native Linux"
fi

if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
    HAS_GPU=true
else
    warn "No GPU detected (nvidia-smi not found)"
    HAS_GPU=false
fi

# ======================== 2. Install base dependencies ========================
log "2. Installing base dependencies"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    git make build-essential gfortran \
    openmpi-bin libopenmpi-dev \
    linux-tools-common \
    python3 python3-pip \
    wget curl cmake jq 2>/dev/null

pip3 install openai anthropic --break-system-packages 2>/dev/null || \
    pip3 install openai anthropic

# ======================== 3. NVIDIA HPC SDK ========================
log "3. Checking NVIDIA HPC SDK"

if command -v nvfortran &>/dev/null; then
    echo "  nvfortran installed: $(nvfortran --version | head -1)"
else
    log "  Installing NVIDIA HPC SDK..."
    curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg 2>/dev/null || true
    echo "deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] \
        https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" \
        | sudo tee /etc/apt/sources.list.d/nvhpc.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvhpc 2>/dev/null || warn "Auto-install failed. Download manually from https://developer.nvidia.com/hpc-sdk-downloads"

    NVHPC_PATH=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin 2>/dev/null | tail -1)
    if [ -n "$NVHPC_PATH" ]; then
        echo "export PATH=$NVHPC_PATH:\$PATH" >> ~/.bashrc
        export PATH="$NVHPC_PATH:$PATH"
    fi
fi

# ======================== 4. Create directory structure ========================
log "4. Creating project directories"

mkdir -p "$PROJECT_ROOT"/{scripts,apps,docs,patches}

# ======================== 5. Clone and set up applications ========================
log "5. Setting up applications from app_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    err "app_config.json not found at $CONFIG_FILE"
    exit 1
fi

# Helper: read config with Python (no jq dependency required, but faster with jq)
cfg_py() {
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    data = json.load(f)
keys = '$2'.split('.')
node = data['$1']
for k in keys:
    if isinstance(node, dict):
        node = node.get(k, '')
    elif isinstance(node, list):
        node = node[int(k)] if k.isdigit() else ''
    else:
        node = ''
        break
if isinstance(node, list):
    print(' '.join(str(x) for x in node))
elif isinstance(node, bool):
    print('true' if node else 'false')
else:
    print(node)
"
}

# Get list of all app names
APP_NAMES=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    data = json.load(f)
for k in data:
    if not k.startswith('_'):
        print(k)
")

for app in $APP_NAMES; do
    # Skip if not target
    if [ "$TARGET_APP" != "all" ] && [ "$TARGET_APP" != "$app" ]; then
        continue
    fi

    log "  Setting up: $app"

    repo_url=$(cfg_py "$app" "repo_url")
    src_subdir=$(cfg_py "$app" "src_subdir")

    # Install app-specific dependencies
    deps=$(cfg_py "$app" "build.dependencies" 2>/dev/null || echo "")
    if [ -n "$deps" ] && [ "$deps" != "None" ]; then
        echo "    Installing dependencies: $deps"
        sudo apt-get install -y -qq $deps 2>/dev/null || warn "Some deps may need manual install"
    fi

    # Clone source
    app_dir="$PROJECT_ROOT/apps/$app"
    if [ ! -d "$app_dir" ]; then
        if [ -n "$repo_url" ] && [ "$repo_url" != "None" ]; then
            echo "    Cloning from $repo_url..."
            git clone "$repo_url" "$app_dir" 2>/dev/null || {
                warn "Clone failed for $app"
                continue
            }
        else
            mkdir -p "$app_dir/$src_subdir"
            echo "    No repo URL configured, created empty directory"
        fi
    else
        echo "    $app already exists, skipping clone"
    fi

    # Create GPU copy
    gpu_dir="$PROJECT_ROOT/apps/${app}_gpu"
    if [ ! -d "$gpu_dir" ]; then
        cp -r "$app_dir" "$gpu_dir"
        mkdir -p "$gpu_dir/original_backup"

        # Determine source file extensions
        src_exts=$(cfg_py "$app" "gpu_patch.source_extensions" 2>/dev/null || echo ".c .h .F")
        for ext in $src_exts; do
            # Remove leading dot for find
            ext_clean="${ext#.}"
            find "$gpu_dir/$src_subdir/" -name "*.$ext_clean" -exec cp {} "$gpu_dir/original_backup/" \; 2>/dev/null || true
        done
        echo "    GPU copy created with backup"
    else
        echo "    GPU copy already exists, skipping"
    fi

    # Create report directory
    mkdir -p "$PROJECT_ROOT/reports/$app/performance_comparison"
done

# ======================== 6. Create .env template ========================
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cat > "$PROJECT_ROOT/.env" << 'ENVFILE'
# API Keys — DO NOT commit to Git!
# Usage: source .env && python3 scripts/02_analyze_hotspots.py <app_name>
export OPENAI_API_KEY="sk-your-key-here"
# export ANTHROPIC_API_KEY="sk-ant-your-key-here"
ENVFILE
    echo "  Created .env template"
fi

# ======================== 7. Create .gitignore ========================
cat > "$PROJECT_ROOT/.gitignore" << 'GITIGNORE'
# API keys
.env
API.txt

# Build artifacts
*.o
*.mod
*.x
*_build/

# Profiling data
gmon.out
perf.data*

# Large files
*.log

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.pyc
GITIGNORE

# ======================== Summary ========================
log "=========================================="
log " Setup complete!"
log "=========================================="
echo ""
echo "  Project root: $PROJECT_ROOT"
echo "  Config file:  $CONFIG_FILE"
echo ""
echo "  Applications set up:"
for app in $APP_NAMES; do
    if [ "$TARGET_APP" != "all" ] && [ "$TARGET_APP" != "$app" ]; then
        continue
    fi
    lang=$(cfg_py "$app" "language")
    domain=$(cfg_py "$app" "domain")
    echo "    $app ($lang) — $domain"
done
echo ""
echo "  Next steps:"
echo "    1. Edit API key:  nano $PROJECT_ROOT/.env"
echo "    2. Profile an app: bash scripts/01_profile_app.sh <app_name>"
echo "    3. LLM analysis:  source .env && python3 scripts/02_analyze_hotspots.py <app_name>"
echo "    4. GPU patching:  python3 scripts/03_apply_gpu_patch.py <app_name>"
echo ""
if [ "$HAS_GPU" = true ]; then
    echo "  GPU detected ✓"
else
    echo "  GPU not detected ✗ — check driver installation"
fi