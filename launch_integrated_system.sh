#!/bin/bash
# Launcher script for Integrated Autonomous Vehicle System v1.3.0

echo "================================="
echo "Integrated AV System Launcher"
echo "Version 1.3.0"
echo "================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "It's recommended to run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Launch the integrated system
echo ""
echo "Launching Integrated AV System..."
echo ""

python3 integrated_av_system.py

echo ""
echo "System shut down."
