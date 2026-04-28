#!/bin/bash

# Exit on error to prevent cascading failures
set -e

echo "============================================================"
echo " Starting setup script for Python environment and Unsloth"
echo "============================================================"

# Ensure ROCm binaries are in PATH and set ROCM_PATH for bitsandbytes/flash-attn permanently
echo ""
echo "Configuring ROCm environment variables in ~/.bashrc..."
if ! grep -q "ROCM_PATH" ~/.bashrc; then
    echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/llvm/bin' >> ~/.bashrc
    echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
    echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc
fi
# Also export them for the current script execution
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/llvm/bin
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

echo ""
echo "[1/8] Installing Python venv module..."
# Attempting to install for both python3.12 and native python3
sudo apt-get update
sudo apt-get install -y python3-venv python3.12-venv git curl

echo ""
echo "[2/8] Setting PIP_BREAK_SYSTEM_PACKAGES for Ubuntu..."
# This environment variable bypasses the externally-managed-environment error (PEP 668)
export PIP_BREAK_SYSTEM_PACKAGES=1

echo ""
echo "[3/8] Initializing venv (preferring Python 3.12)..."
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "Found python3.12"
else
    PYTHON_CMD="python3"
    echo "Python 3.12 not found. Falling back to $PYTHON_CMD"
fi

$PYTHON_CMD -m venv venv
# Activate the virtual environment
source venv/bin/activate

echo ""
echo "[4/8] Installing Astral uv..."
# Installing uv directly into the virtual environment using pip is often cleaner,
# but we can also use the standalone installer. We will use the standalone installer 
# and ensure it's in the PATH, but also install via pip to guarantee it works in venv.
curl -LsSf https://astral.sh/uv/install.sh | sh
# Load environment to put uv in PATH if installed globally
source $HOME/.local/bin/env 2>/dev/null || source $HOME/.cargo/env 2>/dev/null || true
# Fallback to pip install in venv just in case it's not in PATH
pip install uv

echo ""
echo "[5/8] Installing Unsloth (AMD/ROCm version)..."
uv pip install unsloth
# Required pre-release/specific build of bitsandbytes for ROCm to avoid 4-bit decoding bugs
pip install --no-deps "bitsandbytes"

echo ""
echo "[6/8] Installing pandas and datasets modules..."
uv pip install pandas datasets

echo ""
echo "[7/8] Force-Installing PyTorch (ROCm 7.2)..."
# We do this LAST and with --force-reinstall to guarantee that the standard PyPI 'torch'
# installed by Unsloth/dependencies is completely replaced by the ROCm-specific version.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 --upgrade --force-reinstall

echo ""
echo "[8/8] Cloning git repository EIC..."
if [ -d "EIC" ]; then
    echo "Directory 'EIC' already exists. Skipping clone."
else
    git clone https://github.com/noob-master-jpb/EIC.git
fi

echo ""
echo "============================================================"
echo " Setup completed successfully!"
echo "============================================================"
echo "To activate your environment and variables in the future, run:"
echo "    source ~/.bashrc"
echo "    source venv/bin/activate"
echo ""
