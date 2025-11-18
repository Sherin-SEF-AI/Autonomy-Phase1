#!/usr/bin/env python3
"""
Main entry point for the Autonomous Vehicle Perception System.

This application provides a professional-grade multi-camera perception platform
for autonomous vehicle development.
"""

import sys
import signal
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import MainWindow
from utils.logger import get_logger
from utils.system_check import run_system_check


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully."""
    logger = get_logger()
    logger.info("Received interrupt signal, shutting down...")
    QApplication.quit()


def main():
    """
    Main application entry point.
    """
    # Setup logging
    log_dir = Path(__file__).parent / "data" / "logs"
    logger = get_logger("AutonomousPerception", log_dir=log_dir)

    logger.info("=" * 80)
    logger.info("AUTONOMOUS VEHICLE PERCEPTION SYSTEM")
    logger.info("Version 1.0.0")
    logger.info("=" * 80)

    # Run system check
    logger.info("Running system check...")
    try:
        system_ok = run_system_check(verbose=False)
        if system_ok:
            logger.info("System check passed ✓")
        else:
            logger.warning("System check found issues (see above)")
            logger.warning("Attempting to continue anyway...")
    except Exception as e:
        logger.warning(f"System check failed: {e}")
        logger.warning("Continuing anyway...")

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)

    # Note: High DPI scaling is enabled by default in PyQt6
    # AA_EnableHighDpiScaling and AA_UseHighDpiPixmaps are no longer needed

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Autonomous Vehicle Perception System")
    app.setOrganizationName("AV Perception")
    app.setOrganizationDomain("av-perception.org")

    # Set application style
    app.setStyle("Fusion")

    # Apply dark theme
    from PyQt6.QtGui import QPalette, QColor

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

    app.setPalette(dark_palette)

    # Additional dark theme stylesheet
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff;
            background-color: #2a82da;
            border: 1px solid white;
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
    """)

    # Create and show main window
    logger.info("Creating main window...")
    main_window = MainWindow()
    main_window.show()

    logger.info("Application started successfully")
    logger.info("Ready for operation")

    # Run application event loop
    exit_code = app.exec()

    logger.info("Application exiting with code: {}".format(exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
