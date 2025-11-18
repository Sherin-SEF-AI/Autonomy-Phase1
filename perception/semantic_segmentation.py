"""
Semantic segmentation for pixel-wise scene understanding.

Classifies each pixel into categories: road, sidewalk, vehicle, person, sky, etc.
Useful for drivable area detection and scene understanding.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from pathlib import Path

from utils.logger import get_logger


logger = get_logger()


class SegmentationClass(Enum):
    """Semantic segmentation classes (Cityscapes-based)."""
    ROAD = 0
    SIDEWALK = 1
    BUILDING = 2
    WALL = 3
    FENCE = 4
    POLE = 5
    TRAFFIC_LIGHT = 6
    TRAFFIC_SIGN = 7
    VEGETATION = 8
    TERRAIN = 9
    SKY = 10
    PERSON = 11
    RIDER = 12
    CAR = 13
    TRUCK = 14
    BUS = 15
    TRAIN = 16
    MOTORCYCLE = 17
    BICYCLE = 18
    UNKNOWN = 255


# Color palette for visualization (BGR format)
CLASS_COLORS = {
    SegmentationClass.ROAD: (128, 64, 128),
    SegmentationClass.SIDEWALK: (244, 35, 232),
    SegmentationClass.BUILDING: (70, 70, 70),
    SegmentationClass.WALL: (102, 102, 156),
    SegmentationClass.FENCE: (190, 153, 153),
    SegmentationClass.POLE: (153, 153, 153),
    SegmentationClass.TRAFFIC_LIGHT: (250, 170, 30),
    SegmentationClass.TRAFFIC_SIGN: (220, 220, 0),
    SegmentationClass.VEGETATION: (107, 142, 35),
    SegmentationClass.TERRAIN: (152, 251, 152),
    SegmentationClass.SKY: (70, 130, 180),
    SegmentationClass.PERSON: (220, 20, 60),
    SegmentationClass.RIDER: (255, 0, 0),
    SegmentationClass.CAR: (0, 0, 142),
    SegmentationClass.TRUCK: (0, 0, 70),
    SegmentationClass.BUS: (0, 60, 100),
    SegmentationClass.TRAIN: (0, 80, 100),
    SegmentationClass.MOTORCYCLE: (0, 0, 230),
    SegmentationClass.BICYCLE: (119, 11, 32),
    SegmentationClass.UNKNOWN: (0, 0, 0)
}


class SemanticSegmenter:
    """
    Semantic segmentation using deep learning models.

    Supports DeepLabV3, FCN, or simple color-based segmentation.
    """

    def __init__(
        self,
        model_type: str = "simple",
        model_path: Optional[Path] = None,
        enable_gpu: bool = True,
        input_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize semantic segmenter.

        Args:
            model_type: "deeplabv3", "fcn", or "simple"
            model_path: Optional custom model path
            enable_gpu: Use GPU if available
            input_size: Input resolution for neural models
        """
        self.model_type = model_type
        self.model_path = model_path
        self.enable_gpu = enable_gpu
        self.input_size = input_size

        # Model components
        self.model = None
        self.device = None

        # Statistics
        self.total_segmentations = 0
        self.avg_processing_time_ms = 0.0

        # Initialize model
        self._initialize_model()

        logger.info(f"Semantic segmenter initialized (model: {model_type})")

    def _initialize_model(self):
        """Initialize segmentation model."""
        try:
            if self.model_type == "simple":
                logger.info("Using simple color-based segmentation")
                return

            # Try to import torch for deep learning models
            try:
                import torch
                import torchvision
            except ImportError:
                logger.warning(
                    "PyTorch not available. Falling back to simple segmentation. "
                    "Install with: pip install torch torchvision"
                )
                self.model_type = "simple"
                return

            # Set device
            if self.enable_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for segmentation")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for segmentation")

            # Load model
            if self.model_type == "deeplabv3":
                self._load_deeplabv3()
            elif self.model_type == "fcn":
                self._load_fcn()

        except Exception as e:
            logger.error(f"Failed to initialize segmentation model: {e}")
            logger.info("Falling back to simple segmentation")
            self.model_type = "simple"
            self.model = None

    def _load_deeplabv3(self):
        """Load DeepLabV3 model."""
        try:
            import torch
            import torchvision

            # Load pretrained DeepLabV3
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Loaded DeepLabV3 model")

        except Exception as e:
            logger.error(f"Failed to load DeepLabV3: {e}")
            raise

    def _load_fcn(self):
        """Load FCN model."""
        try:
            import torch
            import torchvision

            # Load pretrained FCN
            self.model = torchvision.models.segmentation.fcn_resnet101(
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Loaded FCN model")

        except Exception as e:
            logger.error(f"Failed to load FCN: {e}")
            raise

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform semantic segmentation.

        Args:
            image: BGR image

        Returns:
            Segmentation mask (H x W) with class IDs
        """
        import time
        start_time = time.time()

        if self.model_type == "simple":
            mask = self._simple_segmentation(image)
        else:
            mask = self._neural_segmentation(image)

        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self.total_segmentations += 1

        alpha = 0.1
        self.avg_processing_time_ms = (
            alpha * processing_time +
            (1 - alpha) * self.avg_processing_time_ms
        )

        return mask

    def _neural_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Segment using neural network."""
        try:
            import torch
            import torchvision.transforms as T

            # Preprocess
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Transform
            input_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                mask = output.argmax(0).byte().cpu().numpy()

            # Resize to original size
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            return mask

        except Exception as e:
            logger.error(f"Neural segmentation failed: {e}")
            return self._simple_segmentation(image)

    def _simple_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Simple color-based segmentation (fallback).
        Very basic and inaccurate - for demonstration only.
        """
        h, w = image.shape[:2]
        mask = np.full((h, w), SegmentationClass.UNKNOWN.value, dtype=np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Simple heuristics
        # Sky (top third, blue-ish)
        sky_region = hsv[:h//3, :]
        sky_mask = cv2.inRange(sky_region, np.array([90, 20, 100]), np.array([130, 255, 255]))
        mask[:h//3, :][sky_mask > 0] = SegmentationClass.SKY.value

        # Road (bottom third, gray-ish)
        road_region = image[2*h//3:, :]
        road_gray = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        road_mask = cv2.inRange(road_gray, 30, 120)
        mask[2*h//3:, :][road_mask > 0] = SegmentationClass.ROAD.value

        # Vegetation (green)
        veg_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask[veg_mask > 0] = SegmentationClass.VEGETATION.value

        return mask

    def create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create colored visualization of segmentation mask.

        Args:
            mask: Segmentation mask with class IDs

        Returns:
            BGR colored mask
        """
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for seg_class in SegmentationClass:
            color = CLASS_COLORS.get(seg_class, (0, 0, 0))
            colored[mask == seg_class.value] = color

        return colored

    def overlay_on_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay segmentation on original image.

        Args:
            image: Original BGR image
            mask: Segmentation mask
            alpha: Transparency (0=only image, 1=only mask)

        Returns:
            Blended image
        """
        colored_mask = self.create_colored_mask(mask)

        # Resize if needed
        if colored_mask.shape[:2] != image.shape[:2]:
            colored_mask = cv2.resize(
                colored_mask,
                (image.shape[1], image.shape[0])
            )

        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return overlay

    def extract_drivable_area(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract drivable area (road) from segmentation mask.

        Args:
            mask: Segmentation mask

        Returns:
            Binary mask of drivable area
        """
        drivable = np.zeros_like(mask, dtype=np.uint8)
        drivable[mask == SegmentationClass.ROAD.value] = 255

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, kernel)
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_OPEN, kernel)

        return drivable

    def get_class_percentages(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate percentage of each class in the mask.

        Args:
            mask: Segmentation mask

        Returns:
            Dictionary mapping class names to percentages
        """
        total_pixels = mask.size
        percentages = {}

        for seg_class in SegmentationClass:
            count = np.sum(mask == seg_class.value)
            percentage = (count / total_pixels) * 100
            if percentage > 0.1:  # Only include if > 0.1%
                percentages[seg_class.name.lower()] = percentage

        return percentages

    def detect_obstacles_on_road(
        self,
        mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Detect obstacles on the road (non-road pixels in road area).

        Args:
            mask: Segmentation mask

        Returns:
            List of (x, y) coordinates of obstacles
        """
        # Get road mask
        road_mask = (mask == SegmentationClass.ROAD.value).astype(np.uint8)

        # Detect obstacles (persons, vehicles on road)
        obstacle_classes = [
            SegmentationClass.PERSON.value,
            SegmentationClass.CAR.value,
            SegmentationClass.TRUCK.value,
            SegmentationClass.BUS.value,
            SegmentationClass.MOTORCYCLE.value,
            SegmentationClass.BICYCLE.value
        ]

        obstacle_points = []
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if road_mask[y, x] > 0 and mask[y, x] in obstacle_classes:
                    obstacle_points.append((x, y))

        return obstacle_points

    def get_statistics(self) -> Dict:
        """Get segmentation statistics."""
        return {
            "model_type": self.model_type,
            "total_segmentations": self.total_segmentations,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "device": str(self.device) if self.device else "none",
            "gpu_enabled": self.enable_gpu
        }
