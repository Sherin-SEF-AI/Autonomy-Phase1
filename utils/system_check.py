"""
System check utilities for verifying the environment is properly configured.

Checks for required dependencies, camera access, and directory structure.
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """
    Check Python version meets requirements.

    Returns:
        Tuple of (success, message)
    """
    required_version = (3, 10)
    current_version = sys.version_info[:2]

    if current_version >= required_version:
        return (True, f"Python {current_version[0]}.{current_version[1]} ✓")
    else:
        return (
            False,
            f"Python {required_version[0]}.{required_version[1]}+ required, "
            f"found {current_version[0]}.{current_version[1]}"
        )


def check_required_packages() -> Tuple[bool, List[str]]:
    """
    Check all required packages are installed.

    Returns:
        Tuple of (all_installed, missing_packages)
    """
    required_packages = [
        ('PyQt6', 'PyQt6'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('scipy', 'scipy'),
        ('pyqtgraph', 'pyqtgraph'),
        ('psutil', 'psutil'),
        ('colorlog', 'colorlog'),
    ]

    missing = []
    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)

    return (len(missing) == 0, missing)


def check_camera_access() -> Tuple[bool, str]:
    """
    Check if camera devices can be accessed.

    Returns:
        Tuple of (success, message)
    """
    try:
        import cv2

        # Try to open default camera
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            cap.release()
            return (True, "Camera access ✓")
        else:
            return (False, "Cannot open camera (may be in use or not connected)")

    except Exception as e:
        return (False, f"Camera check failed: {e}")


def check_directory_structure() -> Tuple[bool, List[str]]:
    """
    Check required directories exist.

    Returns:
        Tuple of (all_exist, missing_directories)
    """
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        'camera',
        'perception',
        'visualization',
        'ui',
        'utils',
        'safety',
        'recording',
        'data',
        'config',
        'models',
    ]

    missing = []
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            missing.append(dir_name)

    return (len(missing) == 0, missing)


def check_yolo_model() -> Tuple[bool, str]:
    """
    Check if YOLOv8 model is available.

    Returns:
        Tuple of (success, message)
    """
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'yolov8n.pt'

    if model_path.exists():
        return (True, "YOLOv8 model found ✓")
    else:
        return (True, "YOLOv8 model will be downloaded on first run")


def run_system_check(verbose: bool = True) -> bool:
    """
    Run complete system check.

    Args:
        verbose: Print results to console

    Returns:
        True if all checks pass
    """
    checks_passed = True
    results = []

    # Check Python version
    success, msg = check_python_version()
    results.append(("Python Version", success, msg))
    checks_passed = checks_passed and success

    # Check required packages
    success, missing = check_required_packages()
    if success:
        msg = "All required packages installed ✓"
    else:
        msg = f"Missing packages: {', '.join(missing)}"
    results.append(("Required Packages", success, msg))
    checks_passed = checks_passed and success

    # Check camera access (warning only, not critical)
    success, msg = check_camera_access()
    results.append(("Camera Access", success, msg))
    # Don't fail on camera check - cameras might be added later

    # Check directory structure
    success, missing = check_directory_structure()
    if success:
        msg = "Directory structure OK ✓"
    else:
        msg = f"Missing directories: {', '.join(missing)}"
    results.append(("Directory Structure", success, msg))
    checks_passed = checks_passed and success

    # Check YOLO model (info only)
    success, msg = check_yolo_model()
    results.append(("YOLOv8 Model", success, msg))

    # Print results if verbose
    if verbose:
        print("\n" + "=" * 80)
        print("SYSTEM CHECK RESULTS")
        print("=" * 80)

        for check_name, success, msg in results:
            status = "✓" if success else "✗"
            print(f"{status} {check_name}: {msg}")

        print("=" * 80)

        if checks_passed:
            print("✓ System check passed! Ready to run.")
        else:
            print("✗ System check failed. Please resolve issues above.")
        print("=" * 80 + "\n")

    return checks_passed


if __name__ == "__main__":
    success = run_system_check(verbose=True)
    sys.exit(0 if success else 1)
