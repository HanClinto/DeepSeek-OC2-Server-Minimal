#!/usr/bin/env bash
# install.sh – Set up the DeepSeek-OCR2 server environment.
# Requires: Linux, CUDA-capable GPU, internet access.
set -euo pipefail

# ── 1. Install uv if not present ─────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the remainder of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"

# ── 2. Create venv and install base dependencies ──────────────────────────────
uv sync

# ── 3. Detect CUDA version and install Unsloth ───────────────────────────────
CUDA_VER="121"   # default: CUDA 12.1
TORCH_VER="250"  # default: PyTorch 2.5.0

if command -v nvcc &>/dev/null; then
    raw=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
    if [[ -n "$raw" ]]; then
        CUDA_VER=$(echo "$raw" | tr -d '.')
    fi
elif command -v nvidia-smi &>/dev/null; then
    raw=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
    if [[ -n "$raw" ]]; then
        CUDA_VER=$(echo "$raw" | tr -d '.')
    fi
fi

echo "Detected CUDA version tag: cu${CUDA_VER}"
echo "Installing Unsloth (cu${CUDA_VER}-torch${TORCH_VER})..."

# Try the specific CUDA+Torch combo; fall back to a generic install on failure.
uv add "unsloth[cu${CUDA_VER}-torch${TORCH_VER}] @ git+https://github.com/unslothai/unsloth.git" \
    || uv add "unsloth @ git+https://github.com/unslothai/unsloth.git"

# ── 4. Download the model ─────────────────────────────────────────────────────
echo "Downloading unsloth/DeepSeek-OCR-2 model (this may take a while)..."
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/DeepSeek-OCR-2", local_dir="deepseek_ocr")
print("Model downloaded to ./deepseek_ocr")
EOF

echo ""
echo "✅ Installation complete. Run ./start.sh to start the server."
