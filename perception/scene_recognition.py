"""
Scene recognition for environmental context.

Classifies:
- Weather conditions (sunny, rainy, foggy, night)
- Road type (highway, urban, residential)
- Time of day
- Lighting conditions
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

from utils.logger import get_logger


logger = get_logger()


class WeatherCondition(Enum):
    """Weather conditions."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"
    UNKNOWN = "unknown"


class RoadType(Enum):
    """Road types."""
    HIGHWAY = "highway"
    URBAN_STREET = "urban_street"
    RESIDENTIAL = "residential"
    RURAL = "rural"
    PARKING_LOT = "parking_lot"
    UNKNOWN = "unknown"


class TimeOfDay(Enum):
    """Time of day."""
    DAY = "day"
    DUSK = "dusk"
    NIGHT = "night"
    DAWN = "dawn"
    UNKNOWN = "unknown"


class LightingCondition(Enum):
    """Lighting conditions."""
    BRIGHT = "bright"
    NORMAL = "normal"
    DIM = "dim"
    DARK = "dark"
    UNKNOWN = "unknown"


@dataclass
class SceneContext:
    """Complete scene recognition result."""
    weather: WeatherCondition
    weather_confidence: float

    road_type: RoadType
    road_confidence: float

    time_of_day: TimeOfDay
    time_confidence: float

    lighting: LightingCondition
    lighting_confidence: float

    # Additional metrics
    brightness: float  # 0-255
    contrast: float  # 0-1
    saturation: float  # 0-1
    visibility_score: float  # 0-1, higher is better


class SceneRecognizer:
    """
    Recognizes environmental scene characteristics.

    Uses image statistics, color analysis, and heuristics.
    Can be enhanced with deep learning models.
    """

    def __init__(
        self,
        history_length: int = 30,
        use_temporal_smoothing: bool = True
    ):
        """
        Initialize scene recognizer.

        Args:
            history_length: Number of frames for temporal smoothing
            use_temporal_smoothing: Smooth predictions over time
        """
        self.history_length = history_length
        self.use_temporal_smoothing = use_temporal_smoothing

        # History for temporal smoothing
        self.weather_history: deque = deque(maxlen=history_length)
        self.road_type_history: deque = deque(maxlen=history_length)
        self.time_history: deque = deque(maxlen=history_length)
        self.lighting_history: deque = deque(maxlen=history_length)

        # Statistics
        self.total_recognitions = 0

        logger.info("Scene recognizer initialized")

    def recognize(self, image: np.ndarray) -> SceneContext:
        """
        Recognize scene characteristics.

        Args:
            image: BGR image

        Returns:
            SceneContext with classifications
        """
        # Extract image features
        features = self._extract_features(image)

        # Classify weather
        weather, weather_conf = self._classify_weather(features)

        # Classify road type
        road_type, road_conf = self._classify_road_type(features, image)

        # Classify time of day
        time_of_day, time_conf = self._classify_time_of_day(features)

        # Classify lighting
        lighting, lighting_conf = self._classify_lighting(features)

        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            weather = self._smooth_classification(weather, self.weather_history)
            road_type = self._smooth_classification(road_type, self.road_type_history)
            time_of_day = self._smooth_classification(time_of_day, self.time_history)
            lighting = self._smooth_classification(lighting, self.lighting_history)

        # Calculate visibility score
        visibility = self._calculate_visibility(features, weather)

        context = SceneContext(
            weather=weather,
            weather_confidence=weather_conf,
            road_type=road_type,
            road_confidence=road_conf,
            time_of_day=time_of_day,
            time_confidence=time_conf,
            lighting=lighting,
            lighting_confidence=lighting_conf,
            brightness=features['brightness'],
            contrast=features['contrast'],
            saturation=features['saturation'],
            visibility_score=visibility
        )

        self.total_recognitions += 1

        return context

    def _extract_features(self, image: np.ndarray) -> Dict:
        """Extract image features for classification."""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Brightness (mean value)
        brightness = np.mean(gray)

        # Contrast (standard deviation)
        contrast = np.std(gray) / 128.0  # Normalize to 0-1

        # Saturation (mean saturation in HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0

        # Edge density (how sharp the image is)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Color distribution
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

        # Blur/sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)

        # Sky region analysis (top third)
        sky_region = image[:image.shape[0]//3, :]
        sky_brightness = np.mean(cv2.cvtColor(sky_region, cv2.COLOR_BGR2GRAY))
        sky_saturation = np.mean(cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)[:, :, 1])

        return {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'edge_density': edge_density,
            'sharpness': sharpness,
            'sky_brightness': sky_brightness,
            'sky_saturation': sky_saturation,
            'hist_b': hist_b,
            'hist_g': hist_g,
            'hist_r': hist_r
        }

    def _classify_weather(self, features: Dict) -> Tuple[WeatherCondition, float]:
        """Classify weather condition."""
        brightness = features['brightness']
        saturation = features['saturation']
        contrast = features['contrast']
        sharpness = features['sharpness']

        scores = {}

        # Sunny: High brightness, high saturation, sharp
        scores[WeatherCondition.SUNNY] = (
            (brightness > 120) * 0.4 +
            (saturation > 0.3) * 0.3 +
            (sharpness > 100) * 0.3
        )

        # Cloudy: Medium brightness, low saturation
        scores[WeatherCondition.CLOUDY] = (
            (80 < brightness < 140) * 0.4 +
            (saturation < 0.3) * 0.3 +
            (contrast < 0.6) * 0.3
        )

        # Rainy: Low brightness, very low saturation, blurry
        scores[WeatherCondition.RAINY] = (
            (brightness < 100) * 0.3 +
            (saturation < 0.2) * 0.3 +
            (sharpness < 50) * 0.4
        )

        # Foggy: Very low contrast, medium brightness, very blurry
        scores[WeatherCondition.FOGGY] = (
            (contrast < 0.4) * 0.4 +
            (sharpness < 30) * 0.4 +
            (80 < brightness < 150) * 0.2
        )

        # Get best match
        if max(scores.values()) < 0.3:
            return WeatherCondition.UNKNOWN, 0.0

        best_weather = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_weather], 1.0)

        return best_weather, confidence

    def _classify_road_type(
        self,
        features: Dict,
        image: np.ndarray
    ) -> Tuple[RoadType, float]:
        """Classify road type."""
        edge_density = features['edge_density']
        contrast = features['contrast']

        # Analyze road region (bottom third)
        h = image.shape[0]
        road_region = image[2*h//3:, :]

        # Count horizontal lines (lane markings)
        gray_road = cv2.cvtColor(road_region, cv2.COLOR_BGR2GRAY)
        edges_road = cv2.Canny(gray_road, 50, 150)
        lines = cv2.HoughLinesP(
            edges_road,
            1,
            np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )

        line_count = len(lines) if lines is not None else 0

        scores = {}

        # Highway: High edge density, many lane markings
        scores[RoadType.HIGHWAY] = (
            (edge_density > 0.15) * 0.4 +
            (line_count > 5) * 0.6
        )

        # Urban: Medium edge density, some structure
        scores[RoadType.URBAN_STREET] = (
            (0.1 < edge_density < 0.2) * 0.5 +
            (2 < line_count < 8) * 0.5
        )

        # Residential: Lower edge density, fewer markings
        scores[RoadType.RESIDENTIAL] = (
            (edge_density < 0.12) * 0.5 +
            (line_count < 4) * 0.5
        )

        # Parking lot: Low edge density, very few lines
        scores[RoadType.PARKING_LOT] = (
            (edge_density < 0.08) * 0.7 +
            (line_count < 2) * 0.3
        )

        if max(scores.values()) < 0.3:
            return RoadType.UNKNOWN, 0.0

        best_road = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_road], 1.0)

        return best_road, confidence

    def _classify_time_of_day(self, features: Dict) -> Tuple[TimeOfDay, float]:
        """Classify time of day."""
        brightness = features['brightness']
        sky_brightness = features['sky_brightness']

        scores = {}

        # Day: High brightness
        scores[TimeOfDay.DAY] = min((brightness / 150.0), 1.0) if brightness > 100 else 0.0

        # Night: Very low brightness
        scores[TimeOfDay.NIGHT] = (1.0 - brightness / 100.0) if brightness < 50 else 0.0

        # Dusk/Dawn: Medium brightness
        if 50 < brightness < 100:
            # Dusk tends to have warmer colors
            scores[TimeOfDay.DUSK] = 0.7
            scores[TimeOfDay.DAWN] = 0.5
        elif 100 < brightness < 130:
            scores[TimeOfDay.DUSK] = 0.5
            scores[TimeOfDay.DAWN] = 0.7

        if max(scores.values()) < 0.3:
            return TimeOfDay.UNKNOWN, 0.0

        best_time = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_time], 1.0)

        return best_time, confidence

    def _classify_lighting(self, features: Dict) -> Tuple[LightingCondition, float]:
        """Classify lighting condition."""
        brightness = features['brightness']

        if brightness > 150:
            return LightingCondition.BRIGHT, 0.9
        elif brightness > 100:
            return LightingCondition.NORMAL, 0.8
        elif brightness > 50:
            return LightingCondition.DIM, 0.8
        else:
            return LightingCondition.DARK, 0.9

    def _calculate_visibility(
        self,
        features: Dict,
        weather: WeatherCondition
    ) -> float:
        """Calculate visibility score (0-1, higher is better)."""
        # Base visibility from sharpness and contrast
        base_visibility = (
            min(features['sharpness'] / 200.0, 1.0) * 0.5 +
            features['contrast'] * 0.5
        )

        # Adjust for weather
        weather_penalty = {
            WeatherCondition.SUNNY: 0.0,
            WeatherCondition.CLOUDY: 0.1,
            WeatherCondition.RAINY: 0.3,
            WeatherCondition.FOGGY: 0.5,
            WeatherCondition.SNOWY: 0.4,
            WeatherCondition.UNKNOWN: 0.0
        }

        visibility = max(0.0, base_visibility - weather_penalty.get(weather, 0.0))

        return visibility

    def _smooth_classification(
        self,
        current_class,
        history: deque
    ):
        """Apply temporal smoothing to classification."""
        history.append(current_class)

        if len(history) < 3:
            return current_class

        # Count occurrences
        from collections import Counter
        counts = Counter(history)

        # Return most common
        return counts.most_common(1)[0][0]

    def is_poor_visibility(self, context: SceneContext) -> bool:
        """Check if visibility is poor."""
        return (
            context.visibility_score < 0.4 or
            context.weather in [WeatherCondition.FOGGY, WeatherCondition.RAINY] or
            context.lighting in [LightingCondition.DIM, LightingCondition.DARK]
        )

    def requires_caution(self, context: SceneContext) -> bool:
        """Check if conditions require extra caution."""
        return (
            self.is_poor_visibility(context) or
            context.weather == WeatherCondition.RAINY or
            context.time_of_day in [TimeOfDay.NIGHT, TimeOfDay.DUSK, TimeOfDay.DAWN]
        )

    def get_statistics(self) -> Dict:
        """Get recognition statistics."""
        return {
            "total_recognitions": self.total_recognitions,
            "history_length": self.history_length,
            "temporal_smoothing": self.use_temporal_smoothing
        }
