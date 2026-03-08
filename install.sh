#!/bin/bash
# Install script for nlp_group_project
# Handles the tricky PyTorch Geometric dependencies properly on macOS Apple Silicon
set -e

echo "=========================================="
echo "  NLP Group Project - Dependency Installer"
echo "=========================================="

# Step 1: Install PyTorch 2.5.0 (has confirmed pre-built PyG wheels for macOS)
echo ""
echo "[1/5] Installing PyTorch 2.5.0..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0

# Verify
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "       ✓ Installed PyTorch ${TORCH_VERSION}"

# Step 2: Install PyG extension packages from the official wheel index
# These have pre-built macOS universal2 wheels for cp311 + torch 2.5.0
echo ""
echo "[2/5] Installing PyTorch Geometric extension packages..."
echo "       (downloading pre-built wheels from data.pyg.org)"
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-2.5.0+cpu.html"

echo "       ✓ PyG extensions installed"

# Step 3: Install PyTorch Geometric itself
echo ""
echo "[3/5] Installing PyTorch Geometric..."
pip install torch-geometric

echo "       ✓ PyTorch Geometric installed"

# Step 4: Install remaining requirements
echo ""
echo "[4/5] Installing remaining dependencies..."
pip install -r requirements_base.txt

echo "       ✓ All dependencies installed"

# Step 5: Verify
echo ""
echo "[5/5] Verifying installation..."
python -c "
import torch
import torch_geometric
import torch_scatter
import torch_sparse
print(f'  PyTorch:           {torch.__version__}')
print(f'  PyTorch Geometric: {torch_geometric.__version__}')
print(f'  Device:            {\"MPS\" if torch.backends.mps.is_available() else \"CPU\"}')
print()
print('  ✓ All imports successful!')
"

echo ""
echo "=========================================="
echo "  Installation complete!"
echo "=========================================="
