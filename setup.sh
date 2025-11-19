#!/bin/bash
# Automated setup script for AV Perception System v1.1.0
# This script sets up the environment and dependencies for the ADAS platform

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root/sudo"
    exit 1
fi

print_header "AV Perception System Setup - v1.1.0"
echo ""

# 1. Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ is required. Found: $PYTHON_VERSION"
    print_info "Please install Python 3.10 or higher"
    exit 1
else
    print_success "Python version: $PYTHON_VERSION"
fi

# 2. Check if virtual environment exists
print_info "Checking for virtual environment..."
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment found"
fi

# 3. Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# 4. Upgrade pip
print_info "Upgrading pip..."
python -m pip install --upgrade pip --quiet
print_success "pip upgraded"

# 5. Install dependencies
print_info "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# 6. Create necessary directories
print_info "Creating data directories..."
mkdir -p data/recordings
mkdir -p data/recordings/events
mkdir -p data/logs
mkdir -p data/exports
mkdir -p data/analytics
mkdir -p config/camera_profiles
print_success "Directories created"

# 7. Check for camera access (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "Checking camera access permissions..."
    if groups | grep -q video; then
        print_success "User is in 'video' group"
    else
        print_warning "User is NOT in 'video' group"
        print_info "Run: sudo usermod -a -G video $USER"
        print_info "Then log out and log back in"
    fi
fi

# 8. Test imports
print_info "Testing critical imports..."
python3 -c "
import sys
try:
    import cv2
    print('✓ OpenCV imported successfully')
except ImportError as e:
    print('✗ OpenCV import failed:', e)
    sys.exit(1)

try:
    import PyQt6
    print('✓ PyQt6 imported successfully')
except ImportError as e:
    print('✗ PyQt6 import failed:', e)
    sys.exit(1)

try:
    import numpy
    print('✓ NumPy imported successfully')
except ImportError as e:
    print('✗ NumPy import failed:', e)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "All critical imports successful"
else
    print_error "Import test failed"
    exit 1
fi

# 9. Download YOLOv8 model (optional)
print_info "Checking for YOLOv8 model..."
if [ ! -f "yolov8n.pt" ]; then
    print_warning "YOLOv8 model not found. It will be downloaded on first run."
    read -p "Download YOLOv8n model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -c "
from ultralytics import YOLO
print('Downloading YOLOv8n model...')
model = YOLO('yolov8n.pt')
print('✓ YOLOv8n model downloaded')
        "
        print_success "YOLOv8 model ready"
    fi
else
    print_success "YOLOv8 model found"
fi

# 10. Display system information
echo ""
print_header "System Information"
echo ""
print_info "Python: $(python --version)"
print_info "pip: $(pip --version | awk '{print $2}')"
print_info "OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
print_info "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
print_info "PyQt6: $(python -c 'import PyQt6; print(PyQt6.__version__)' 2>/dev/null || echo 'Unknown')"

# 11. Check available cameras (Linux/macOS)
echo ""
print_info "Checking for available cameras..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CAMERA_COUNT=$(ls /dev/video* 2>/dev/null | wc -l)
    if [ $CAMERA_COUNT -gt 0 ]; then
        print_success "Found $CAMERA_COUNT camera device(s)"
        ls /dev/video* | while read camera; do
            echo "  - $camera"
        done
    else
        print_warning "No camera devices found in /dev/"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_info "Camera detection on macOS requires running the application"
fi

# 12. Setup complete
echo ""
print_header "Setup Complete!"
echo ""
echo -e "${GREEN}The AV Perception System is ready to run!${NC}"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: ${BLUE}source venv/bin/activate${NC}"
echo "  2. Run the application: ${BLUE}python main.py${NC}"
echo ""
echo "For more information, see:"
echo "  - README.md - General documentation"
echo "  - QUICKSTART.md - Quick start guide"
echo "  - docs/ADVANCED_FEATURES.md - Advanced features documentation"
echo ""
echo "To run tests:"
echo "  ${BLUE}pytest tests/${NC}"
echo ""
echo "To run advanced features demo:"
echo "  ${BLUE}python examples/advanced_features_demo.py${NC}"
echo ""
print_success "Setup completed successfully!"
echo ""
