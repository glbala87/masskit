#!/bin/bash
# MassKit - One-step setup and install
set -e

echo "============================================"
echo "  MassKit - Setup"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Install
echo ""
echo "Installing masskit..."
cd python
pip install -e ".[viz,dev]" --quiet
cd ..

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Run the toolkit:"
echo "  python3 run_demo.py          Full interactive demo"
echo "  masskit --help                CLI tool"
echo "  pytest python/tests/ -v      Run tests"
echo ""
