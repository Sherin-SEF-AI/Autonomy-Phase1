"""
Object detection module using YOLOv8.

Provides real-time detection of:
- Vehicles (car, truck, bus, motorcycle)
- Pedestrians
- Cyclists
- Traffic signs and lights
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils.data_structures import DetectedObject, ObjectClass
from utils.logger import get_logger


logger = get_logger()


# YOLO class ID to ObjectClass mapping
YOLO_CLASS_MAP = {
    0: (ObjectClass.PERSON, "person"),
    1: (ObjectClass.BICYCLE, "bicycle"),
    2: (ObjectClass.CAR, "car"),
    3: (ObjectClass.MOTORCYCLE, "motorcycle"),
    5: (ObjectClass.BUS, "bus"),
    7: (ObjectClass.TRUCK, "truck"),
    9: (ObjectClass.TRAFFIC_LIGHT, "traffic_light"),
    11: (ObjectClass.STOP_SIGN, "stop_sign"),
}

# Object heights for distance estimation (meters)
OBJECT_HEIGHTS = {
    ObjectClass.PERSON: 1.7,
    ObjectClass.BICYCLE: 1.0,
    ObjectClass.CAR: 1.5,
    ObjectClass.MOTORCYCLE: 1.2,
    ObjectClass.BUS: 3.0,
    ObjectClass.TRUCK: 3.0,
    ObjectClass.TRAFFIC_LIGHT: 0.3,
    ObjectClass.STOP_SIGN: 0.6,
}


@dataclass
class ObjectDetectionConfig:
    """Configuration for object detection."""
    # Model settings
    model_path: str = "models/yolov8n.pt"
    input_size: int = 640
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "cpu"  # "cpu" or "cuda"

    # Detection interval (run detection every N frames)
    detection_interval: int = 2

    # Classes to detect (YOLO class IDs)
    classes: List[int] = None

    # Distance estimation
    camera_focal_length: float = 500.0  # pixels (will be updated from calibration)

    def __post_init__(self):
        if self.classes is None:
            # Default: detect vehicles, pedestrians, cyclists, signs
            self.classes = [0, 1, 2, 3, 5, 7, 9, 11]


class ObjectDetector:
    """
    YOLOv8-based object detector for autonomous vehicle perception.
    """

    def __init__(self, config: Optional[ObjectDetectionConfig] = None):
        """
        Initialize object detector.

        Args:
            config: Detection configuration
        """
        self.config = config or ObjectDetectionConfig()
        self.model = None
        self.model_loaded = False

        # Frame counter for detection interval
        self.frame_count = 0

        # Cached detections (for frames where we don't run detection)
        self.cached_detections: List[DetectedObject] = []

        # Statistics
        self.total_detections = 0
        self.frames_processed = 0

        # Try to load model
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            model_path = Path(self.config.model_path)

            # Download model if it doesn't exist
            if not model_path.exists():
                logger.info(f"YOLOv8 model not found at {model_path}, downloading...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # ultralytics will auto-download when we create the model
                self.model = YOLO("yolov8n.pt")
                logger.info(f"YOLOv8 model downloaded successfully")
            else:
                self.model = YOLO(str(model_path))
                logger.info(f"YOLOv8 model loaded from {model_path}")

            self.model_loaded = True

        except ImportError:
            logger.error("ultralytics package not installed. Object detection disabled.")
            logger.error("Install with: pip install ultralytics")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model_loaded = False

    def detect(
        self,
        image: np.ndarray,
        camera_id: int,
        timestamp: float
    ) -> List[DetectedObject]:
        """
        Detect objects in an image.

        Args:
            image: Input image (BGR format)
            camera_id: Camera identifier
            timestamp: Frame timestamp

        Returns:
            List of detected objects
        """
        self.frames_processed += 1

        # Check if model is loaded
        if not self.model_loaded or self.model is None:
            return []

        # Run detection based on interval
        if self.frame_count % self.config.detection_interval == 0:
            detections = self._run_detection(image, camera_id, timestamp)
            self.cached_detections = detections
        else:
            # Use cached detections
            detections = self.cached_detections

        self.frame_count += 1

        return detections

    def _run_detection(
        self,
        image: np.ndarray,
        camera_id: int,
        timestamp: float
    ) -> List[DetectedObject]:
        """Run YOLOv8 detection on image."""
        try:
            # Run inference
            results = self.model.predict(
                image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.classes,
                max_det=self.config.max_detections,
                device=self.config.device,
                verbose=False
            )

            # Parse results
            detections = []

            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)

                    # Get class and confidence
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())

                    # Map YOLO class to our ObjectClass
                    if class_id in YOLO_CLASS_MAP:
                        obj_class, class_name = YOLO_CLASS_MAP[class_id]

                        # Estimate distance
                        distance = self._estimate_distance(
                            bbox=(x1, y1, x2, y2),
                            object_class=obj_class,
                            image_height=image.shape[0]
                        )

                        # Create detection object
                        detection = DetectedObject(
                            object_id=None,  # Will be assigned by tracker
                            class_id=obj_class.value,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            camera_id=camera_id,
                            timestamp=timestamp,
                            distance=distance,
                            metadata={
                                "yolo_class_id": class_id
                            }
                        )

                        detections.append(detection)
                        self.total_detections += 1

            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []

    def _estimate_distance(
        self,
        bbox: Tuple[int, int, int, int],
        object_class: ObjectClass,
        image_height: int
    ) -> Optional[float]:
        """
        Estimate distance to object using bounding box height.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            object_class: Object class
            image_height: Image height in pixels

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1

        if bbox_height <= 0:
            return None

        # Get typical object height
        if object_class not in OBJECT_HEIGHTS:
            return None

        real_height = OBJECT_HEIGHTS[object_class]

        # Simple pinhole camera model: distance = (real_height * focal_length) / pixel_height
        distance = (real_height * self.config.camera_focal_length) / bbox_height

        # Clamp to reasonable range (0.5m to 200m)
        distance = np.clip(distance, 0.5, 200.0)

        return float(distance)

    def update_camera_calibration(self, camera_matrix: np.ndarray):
        """
        Update camera calibration parameters.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        # Extract focal length from camera matrix
        self.config.camera_focal_length = float(camera_matrix[1, 1])
        logger.info(f"Updated camera focal length to {self.config.camera_focal_length:.2f} pixels")

    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        avg_detections_per_frame = (
            self.total_detections / self.frames_processed
            if self.frames_processed > 0 else 0.0
        )

        return {
            "model_loaded": self.model_loaded,
            "frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": avg_detections_per_frame,
            "detection_interval": self.config.detection_interval
        }

    def reset(self):
        """Reset statistics and cached detections."""
        self.frame_count = 0
        self.cached_detections = []
        self.total_detections = 0
        self.frames_processed = 0
        logger.info("Object detector reset")
