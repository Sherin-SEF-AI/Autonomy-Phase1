#!/bin/bash
# Launcher script for FUNCTIONAL Integrated Autonomous Vehicle System v1.3.0

echo "========================================="
echo "Integrated AV System Launcher"
echo "FULLY FUNCTIONAL VERSION"
echo "Version 1.3.0"
echo "========================================="
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

# Check for cameras
echo ""
echo "Checking for connected cameras..."
camera_count=$(ls /dev/video* 2>/dev/null | wc -l)
if [ $camera_count -gt 0 ]; then
    echo "✓ Found $camera_count camera device(s)"
    echo "  System will use REAL camera feeds"
else
    echo "⚠ No cameras detected"
    echo "  System will run in SIMULATION mode with test patterns"
fi

# Launch the functional integrated system
echo ""
echo "Launching FUNCTIONAL Integrated AV System..."
echo ""

python3 integrated_av_system_functional.py

echo ""
echo "System shut down."
