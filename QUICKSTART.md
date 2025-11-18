# Quick Start Guide

Get the Autonomous Vehicle Perception System running in 5 minutes!

## Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **USB Cameras** (up to 4 cameras supported)
- **Linux/macOS/Windows** operating system

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Autonomy-Phase1
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- PyQt6 (GUI framework)
- OpenCV (computer vision)
- YOLOv8/PyTorch (object detection)
- And all other dependencies

**Note**: First installation may take 5-10 minutes due to large packages (PyTorch ~2GB).

## Running the System

### Option 1: Direct Python

```bash
python main.py
```

### Option 2: System Check First

Verify your environment is properly configured:

```bash
python -m utils.system_check
```

Expected output:
```
================================================================================
SYSTEM CHECK RESULTS
================================================================================
✓ Python Version: Python 3.10 ✓
✓ Required Packages: All required packages installed ✓
✓ Camera Access: Camera access ✓
✓ Directory Structure: Directory structure OK ✓
✓ YOLOv8 Model: YOLOv8 model will be downloaded on first run
================================================================================
✓ System check passed! Ready to run.
================================================================================
```

Then run the application:

```bash
python main.py
```

## First Run

On first run, the system will:

1. **Download YOLOv8 model** (~6MB) - happens automatically
2. **Create necessary directories** (data/, logs/, recordings/)
3. **Detect available cameras**
4. **Initialize the perception pipeline**

Expected console output:

```
INFO     [AutonomousPerception] ================================================================================
INFO     [AutonomousPerception] AUTONOMOUS VEHICLE PERCEPTION SYSTEM
INFO     [AutonomousPerception] Version 1.0.1
INFO     [AutonomousPerception] ================================================================================
INFO     [AutonomousPerception] Running system check...
INFO     [AutonomousPerception] System check passed ✓
INFO     [AutonomousPerception] Creating main window...
INFO     [AutonomousPerception] Application started successfully
INFO     [AutonomousPerception] Ready for operation
```

## Basic Usage

### Starting Perception

1. **Connect Cameras**: Plug in USB cameras (1-4 cameras)
2. **Launch Application**: Run `python main.py`
3. **Start System**: Click "Start Perception System" button
4. **Watch Results**: Real-time detection, tracking, and lane analysis

### Camera Setup

- **Camera 0**: Laptop/dashboard camera (default)
- **Camera 1**: Front-facing camera
- **Camera 2**: Left-side camera
- **Camera 3**: Right-side camera

To discover cameras:
- Menu → Cameras → Discover Cameras

## Troubleshooting

### "No cameras found"

**Solutions:**
- Ensure USB cameras are plugged in
- Check camera permissions (Linux: add user to `video` group)
- Try different USB ports (prefer USB 3.0)
- Run system check: `python -m utils.system_check`

### "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Low FPS / Performance Issues

**Solutions:**
- Reduce camera resolution (Settings → 640x480)
- Disable unused cameras
- Close other applications
- Increase detection interval (process every 3-5 frames)

### YOLOv8 model download fails

**Manual download:**
```bash
mkdir -p models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Next Steps

### Calibrate Cameras (Recommended)

For accurate distance estimation:

1. Print a chessboard pattern (9x6 or 10x7)
2. Menu → Cameras → Calibrate Camera
3. Follow the wizard (capture 15-20 images)
4. Save calibration data

### Record a Session

1. Menu → Recording → Start Recording
2. Drive/test your scenario
3. Menu → Recording → Stop Recording
4. Playback: Menu → Recording → Open Recorded Session

### Enable Safety Systems

- Menu → Safety → Enable FCW (Forward Collision Warning)
- Menu → Safety → Enable LDW (Lane Departure Warning)
- Menu → Safety → Enable BSW (Blind Spot Warning)

### View Telemetry

- Menu → View → Telemetry Dashboard
- Monitor FPS, latency, CPU, memory in real-time

## Documentation

- **Complete User Manual**: `docs/USER_MANUAL.md` (50+ pages)
- **Integration Guide**: `docs/PHASE4_INTEGRATION.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Changelog**: `CHANGELOG.md`

## Testing

Run tests to verify installation:

```bash
# Unit tests
pytest tests/test_data_structures.py -v

# Safety system tests
pytest tests/test_safety_systems.py -v

# Performance benchmarks
pytest tests/test_performance.py --benchmark-only -v
```

## Support

- **Issues**: Check troubleshooting section in USER_MANUAL.md
- **System Check**: Run `python -m utils.system_check` for diagnostics
- **Logs**: Check `data/logs/` for detailed error messages

## Version

Current Version: **1.0.1**
- ✅ All 5 phases complete
- ✅ Production ready
- ✅ Fully tested

---

**Happy Perceiving! 🚗💨**

For detailed documentation, see `docs/USER_MANUAL.md`.
