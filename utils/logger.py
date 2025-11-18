"""
Logging utilities for the autonomous vehicle perception system.

Provides structured logging with color-coding and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import colorlog


class PerceptionLogger:
    """
    Custom logger for the perception system with color output and file logging.
    """

    def __init__(self, name: str = "AutonomousPerception", log_dir: Optional[Path] = None):
        """
        Initialize the logger.

        Args:
            name: Logger name
            log_dir: Directory for log files (default: data/logs)
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(blue)s[%(name)s]%(reset)s %(message)s',
            datefmt='%H:%M:%S',
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log directory specified)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)

            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.logger.info(f"Logging to file: {log_file}")

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def exception(self, message: str):
        """Log exception with traceback."""
        self.logger.exception(message)


# Global logger instance
_global_logger: Optional[PerceptionLogger] = None


def get_logger(name: str = "AutonomousPerception", log_dir: Optional[Path] = None) -> PerceptionLogger:
    """
    Get or create the global logger instance.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        PerceptionLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = PerceptionLogger(name=name, log_dir=log_dir)
    return _global_logger
