"""
Camera calibration module for intrinsic and extrinsic calibration.

Provides tools for calibrating camera parameters using chessboard patterns.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json
from utils.logger import get_logger


logger = get_logger()


class CameraCalibrator:
    """
    Handles camera calibration using chessboard patterns.
    """

    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6), square_size: float = 0.025):
        """
        Initialize camera calibrator.

        Args:
            chessboard_size: Number of internal corners (columns, rows)
            square_size: Size of chessboard square in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Storage for calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.calibration_images = []

        # Calibration results
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add an image for calibration.

        Args:
            image: Calibration image (must contain chessboard pattern)

        Returns:
            True if chessboard was found, False otherwise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Prepare object points (3D coordinates of chessboard corners)
            objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[
                0:self.chessboard_size[0],
                0:self.chessboard_size[1]
            ].T.reshape(-1, 2)
            objp *= self.square_size

            self.object_points.append(objp)
            self.image_points.append(corners_refined)
            self.calibration_images.append(image.copy())

            logger.info(f"Added calibration image {len(self.calibration_images)}")
            return True
        else:
            logger.warning("Chessboard not found in image")
            return False

    def calibrate(self, image_size: Tuple[int, int]) -> bool:
        """
        Perform camera calibration.

        Args:
            image_size: Image size (width, height)

        Returns:
            True if calibration successful
        """
        if len(self.object_points) < 10:
            logger.error(f"Need at least 10 calibration images, have {len(self.object_points)}")
            return False

        logger.info(f"Calibrating camera with {len(self.object_points)} images...")

        # Perform calibration
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            image_size,
            None,
            None
        )

        if ret:
            self.camera_matrix = camera_matrix
            self.distortion_coeffs = distortion_coeffs
            self.rvecs = rvecs
            self.tvecs = tvecs

            # Calculate reprojection error
            self.calibration_error = self._calculate_reprojection_error()

            logger.info(f"Calibration successful! Reprojection error: {self.calibration_error:.4f} pixels")
            logger.info(f"Camera matrix:\n{self.camera_matrix}")
            logger.info(f"Distortion coefficients: {self.distortion_coeffs.ravel()}")

            return True
        else:
            logger.error("Calibration failed")
            return False

    def _calculate_reprojection_error(self) -> float:
        """
        Calculate mean reprojection error.

        Returns:
            Mean reprojection error in pixels
        """
        total_error = 0
        total_points = 0

        for i in range(len(self.object_points)):
            # Project object points using calibration results
            projected_points, _ = cv2.projectPoints(
                self.object_points[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.distortion_coeffs
            )

            # Calculate error
            error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2)
            total_error += error
            total_points += len(self.object_points[i])

        mean_error = total_error / total_points
        return mean_error

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Undistort an image using calibration results.

        Args:
            image: Distorted image

        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            logger.warning("Camera not calibrated, returning original image")
            return image

        undistorted = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        return undistorted

    def save_calibration(self, filepath: str):
        """
        Save calibration results to file.

        Args:
            filepath: Path to save calibration data (JSON format)
        """
        if self.camera_matrix is None:
            logger.error("No calibration data to save")
            return

        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "calibration_error": float(self.calibration_error),
            "chessboard_size": self.chessboard_size,
            "square_size": self.square_size,
            "num_images": len(self.calibration_images)
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        logger.info(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration results from file.

        Args:
            filepath: Path to calibration data file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)

            self.camera_matrix = np.array(calibration_data["camera_matrix"])
            self.distortion_coeffs = np.array(calibration_data["distortion_coeffs"])
            self.calibration_error = calibration_data["calibration_error"]

            logger.info(f"Calibration loaded from {filepath}")
            logger.info(f"Reprojection error: {self.calibration_error:.4f} pixels")

            return True

        except Exception as e:
            logger.error(f"Failed to load calibration from {filepath}: {e}")
            return False

    def reset(self):
        """Reset calibration data."""
        self.object_points.clear()
        self.image_points.clear()
        self.calibration_images.clear()
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        logger.info("Calibration data reset")

    def get_calibration_quality(self) -> str:
        """
        Get a qualitative assessment of calibration quality.

        Returns:
            Quality rating: "Excellent", "Good", "Fair", or "Poor"
        """
        if self.calibration_error is None:
            return "Not calibrated"

        if self.calibration_error < 0.3:
            return "Excellent"
        elif self.calibration_error < 0.5:
            return "Good"
        elif self.calibration_error < 1.0:
            return "Fair"
        else:
            return "Poor"
