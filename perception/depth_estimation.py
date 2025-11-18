"""
Monocular depth estimation for 3D scene understanding.

Estimates depth from single camera images using deep learning.
Supports multiple models: MiDaS, DPT, custom models.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum
from pathlib import Path

from utils.logger import get_logger


logger = get_logger()


class DepthModel(Enum):
    """Available depth estimation models."""
    MIDAS_SMALL = "midas_small"  # Fast, lower accuracy
    MIDAS_V21 = "midas_v21"  # Balanced
    DPT_HYBRID = "dpt_hybrid"  # High accuracy, slower
    SIMPLE = "simple"  # Simple disparity-based (fallback)


class DepthEstimator:
    """
    Monocular depth estimation using pre-trained deep learning models.

    Note: Requires torch and timm packages for MiDaS/DPT models.
    Falls back to simple stereo-like estimation if not available.
    """

    def __init__(
        self,
        model_type: DepthModel = DepthModel.SIMPLE,
        model_path: Optional[Path] = None,
        enable_gpu: bool = True,
        input_size: Tuple[int, int] = (384, 384)
    ):
        """
        Initialize depth estimator.

        Args:
            model_type: Type of depth model to use
            model_path: Optional path to custom model weights
            enable_gpu: Use GPU if available
            input_size: Input size for neural network models
        """
        self.model_type = model_type
        self.model_path = model_path
        self.enable_gpu = enable_gpu
        self.input_size = input_size

        # Model components
        self.model = None
        self.transform = None
        self.device = None

        # Statistics
        self.total_estimations = 0
        self.avg_processing_time_ms = 0.0

        # Initialize model
        self._initialize_model()

        logger.info(f"Depth estimator initialized (model: {model_type.value})")

    def _initialize_model(self):
        """Initialize the depth estimation model."""
        try:
            if self.model_type == DepthModel.SIMPLE:
                # Simple baseline - no deep learning required
                logger.info("Using simple depth estimation (no ML model)")
                return

            # Try to import torch for deep learning models
            try:
                import torch
            except ImportError:
                logger.warning(
                    "PyTorch not available. Falling back to simple depth estimation. "
                    "Install with: pip install torch torchvision"
                )
                self.model_type = DepthModel.SIMPLE
                return

            # Set device
            if self.enable_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for depth estimation")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for depth estimation")

            # Load model based on type
            if self.model_type in [DepthModel.MIDAS_SMALL, DepthModel.MIDAS_V21]:
                self._load_midas_model()
            elif self.model_type == DepthModel.DPT_HYBRID:
                self._load_dpt_model()

        except Exception as e:
            logger.error(f"Failed to initialize depth model: {e}")
            logger.info("Falling back to simple depth estimation")
            self.model_type = DepthModel.SIMPLE
            self.model = None

    def _load_midas_model(self):
        """Load MiDaS depth estimation model."""
        try:
            import torch

            # Load MiDaS model from torch hub
            if self.model_type == DepthModel.MIDAS_SMALL:
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            else:  # MIDAS_V21
                self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if self.model_type == DepthModel.MIDAS_SMALL:
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded {self.model_type.value} model")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            raise

    def _load_dpt_model(self):
        """Load DPT depth estimation model."""
        try:
            import torch

            # Load DPT model
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform

            self.model.to(self.device)
            self.model.eval()

            logger.info("Loaded DPT Hybrid model")

        except Exception as e:
            logger.error(f"Failed to load DPT model: {e}")
            raise

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from image.

        Args:
            image: BGR image

        Returns:
            Depth map (normalized to 0-255 for visualization, or actual depth values)
        """
        import time
        start_time = time.time()

        if self.model_type == DepthModel.SIMPLE:
            depth_map = self._simple_depth_estimation(image)
        else:
            depth_map = self._neural_depth_estimation(image)

        # Update statistics
        processing_time = (time.time() - start_time) * 1000  # ms
        self.total_estimations += 1

        # Update running average
        alpha = 0.1  # Smoothing factor
        self.avg_processing_time_ms = (
            alpha * processing_time +
            (1 - alpha) * self.avg_processing_time_ms
        )

        return depth_map

    def _neural_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using neural network."""
        try:
            import torch

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            input_batch = self.transform(image_rgb).to(self.device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_map = prediction.cpu().numpy()

            # Normalize to 0-255 for visualization
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            return depth_map

        except Exception as e:
            logger.error(f"Neural depth estimation failed: {e}")
            # Fall back to simple method
            return self._simple_depth_estimation(image)

    def _simple_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Simple depth estimation using edge detection and blur.
        This is a fallback method and not accurate.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Combine blur and edges for pseudo-depth
        # Objects with sharp edges are closer, blurry areas are farther
        depth_map = cv2.addWeighted(blurred, 0.7, edges, 0.3, 0)

        # Invert so closer objects are brighter
        depth_map = 255 - depth_map

        return depth_map

    def get_depth_at_point(
        self,
        depth_map: np.ndarray,
        point: Tuple[int, int],
        radius: int = 5
    ) -> float:
        """
        Get depth value at a specific point (averaged over small region).

        Args:
            depth_map: Depth map from estimate_depth()
            point: (x, y) coordinate
            radius: Averaging radius

        Returns:
            Depth value (0-255 scale, or actual depth if available)
        """
        x, y = point
        h, w = depth_map.shape[:2]

        # Clamp to image bounds
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)

        # Extract region
        region = depth_map[y1:y2, x1:x2]

        # Return mean depth
        return float(np.mean(region))

    def get_depth_for_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """
        Get depth statistics for a bounding box.

        Args:
            depth_map: Depth map
            bbox: (x1, y1, x2, y2)

        Returns:
            Dictionary with min, max, mean, median depth
        """
        x1, y1, x2, y2 = bbox

        # Extract ROI
        roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0
            }

        return {
            "min": float(np.min(roi)),
            "max": float(np.max(roi)),
            "mean": float(np.mean(roi)),
            "median": float(np.median(roi))
        }

    def create_colored_depth_map(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Create colored depth map for visualization.

        Args:
            depth_map: Grayscale depth map
            colormap: OpenCV colormap to use

        Returns:
            BGR colored depth map
        """
        # Ensure 8-bit
        if depth_map.dtype != np.uint8:
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply colormap
        colored = cv2.applyColorMap(depth_map, colormap)

        return colored

    def overlay_depth_on_image(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay colored depth map on original image.

        Args:
            image: Original BGR image
            depth_map: Grayscale depth map
            alpha: Transparency (0=only image, 1=only depth)

        Returns:
            Blended image
        """
        # Create colored depth map
        colored_depth = self.create_colored_depth_map(depth_map)

        # Resize if needed
        if colored_depth.shape[:2] != image.shape[:2]:
            colored_depth = cv2.resize(colored_depth, (image.shape[1], image.shape[0]))

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, colored_depth, alpha, 0)

        return overlay

    def estimate_distance(
        self,
        depth_value: float,
        calibration_factor: float = 1.0
    ) -> float:
        """
        Convert depth value to estimated distance in meters.

        Note: Requires calibration for accurate results.

        Args:
            depth_value: Normalized depth value (0-255)
            calibration_factor: Calibration multiplier

        Returns:
            Estimated distance in meters
        """
        # Simple inverse relationship (requires calibration)
        # This is a rough approximation
        if depth_value < 1:
            return 100.0  # Very far

        # Map 255 (close) -> 1m, 0 (far) -> 100m
        normalized = depth_value / 255.0
        distance = (1.0 - normalized) * 100.0 * calibration_factor

        return max(0.5, min(distance, 100.0))

    def get_statistics(self) -> Dict:
        """Get depth estimation statistics."""
        return {
            "model_type": self.model_type.value,
            "total_estimations": self.total_estimations,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "device": str(self.device) if self.device else "none",
            "gpu_enabled": self.enable_gpu
        }
