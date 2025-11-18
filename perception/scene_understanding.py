"""
Scene understanding for high-level traffic analysis and situational awareness.

Analyzes the overall traffic scene to provide:
- Traffic density and flow patterns
- Scene type classification
- Situational complexity assessment
- Crowd behavior analysis
- Dangerous situation detection
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time

from utils.data_structures import TrackedObject, DetectedObject, LaneDetectionResult
from perception.behavior_classification import ObjectBehavior
from utils.logger import get_logger


logger = get_logger()


class TrafficDensity(Enum):
    """Traffic density levels."""
    EMPTY = "empty"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CONGESTED = "congested"


class SceneType(Enum):
    """Scene type classification."""
    HIGHWAY = "highway"
    URBAN_STREET = "urban_street"
    RESIDENTIAL = "residential"
    PARKING_LOT = "parking_lot"
    INTERSECTION = "intersection"
    UNKNOWN = "unknown"


class TrafficFlow(Enum):
    """Traffic flow direction patterns."""
    SAME_DIRECTION = "same_direction"
    OPPOSITE_DIRECTIONS = "opposite_directions"
    MULTI_DIRECTIONAL = "multi_directional"
    STATIC = "static"
    UNKNOWN = "unknown"


class SituationComplexity(Enum):
    """Situational complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class DangerLevel(Enum):
    """Danger assessment levels."""
    SAFE = "safe"
    CAUTIOUS = "cautious"
    WARNING = "warning"
    DANGER = "danger"


@dataclass
class SceneAnalysis:
    """Complete scene understanding analysis."""
    timestamp: float

    # Traffic metrics
    traffic_density: TrafficDensity
    scene_type: SceneType
    traffic_flow: TrafficFlow
    situation_complexity: SituationComplexity
    danger_level: DangerLevel

    # Quantitative metrics
    object_count: int
    moving_object_count: int
    average_speed: float  # m/s
    speed_variance: float  # Speed diversity
    spatial_density: float  # Objects per square meter

    # Behavioral metrics
    erratic_behavior_count: int  # Objects with unusual behaviors
    collision_risk_count: int  # Objects with collision risk

    # Lane metrics
    lane_keeping_quality: Optional[float] = None  # 0-1, None if no lane data

    # Additional context
    is_intersection: bool = False
    is_crowded: bool = False
    has_pedestrians: bool = False
    has_vulnerable_road_users: bool = False  # Pedestrians, cyclists


class SceneUnderstanding:
    """
    Analyzes the overall traffic scene for high-level understanding.

    Provides situational awareness beyond individual object tracking.
    """

    def __init__(
        self,
        update_interval: float = 1.0,  # seconds
        spatial_range: float = 30.0,  # meters
        history_length: int = 30  # frames
    ):
        """
        Initialize scene understanding.

        Args:
            update_interval: Minimum time between analyses
            spatial_range: Range to consider for spatial density
            history_length: Number of past analyses to keep
        """
        self.update_interval = update_interval
        self.spatial_range = spatial_range
        self.history_length = history_length

        # Analysis history
        self.analysis_history: deque = deque(maxlen=history_length)

        # Timing
        self.last_update_time = 0.0

        logger.info("Scene understanding module initialized")

    def analyze(
        self,
        tracked_objects: List[TrackedObject],
        behaviors: Optional[Dict[int, ObjectBehavior]] = None,
        lane_result: Optional[LaneDetectionResult] = None
    ) -> Optional[SceneAnalysis]:
        """
        Analyze the current traffic scene.

        Args:
            tracked_objects: List of tracked objects
            behaviors: Optional behavior classifications
            lane_result: Optional lane detection result

        Returns:
            SceneAnalysis if update needed, None otherwise
        """
        current_time = time.time()

        # Check if update interval has passed
        if current_time - self.last_update_time < self.update_interval:
            return None

        self.last_update_time = current_time

        # Perform analysis
        analysis = self._perform_analysis(
            tracked_objects, behaviors, lane_result, current_time
        )

        # Store in history
        self.analysis_history.append(analysis)

        return analysis

    def _perform_analysis(
        self,
        tracked_objects: List[TrackedObject],
        behaviors: Optional[Dict[int, ObjectBehavior]],
        lane_result: Optional[LaneDetectionResult],
        timestamp: float
    ) -> SceneAnalysis:
        """Perform complete scene analysis."""

        # Count objects
        object_count = len(tracked_objects)
        moving_count = sum(1 for obj in tracked_objects if not obj.is_stationary)

        # Calculate speed metrics
        speeds = []
        for obj in tracked_objects:
            if obj.current_velocity:
                vx, vy = obj.current_velocity
                speed = np.sqrt(vx**2 + vy**2)
                speeds.append(speed)

        avg_speed = np.mean(speeds) if speeds else 0.0
        speed_variance = np.var(speeds) if len(speeds) > 1 else 0.0

        # Calculate spatial density
        spatial_density = self._calculate_spatial_density(tracked_objects)

        # Classify traffic density
        traffic_density = self._classify_traffic_density(
            object_count, spatial_density
        )

        # Detect scene type
        scene_type = self._classify_scene_type(
            tracked_objects, avg_speed, spatial_density
        )

        # Analyze traffic flow
        traffic_flow = self._analyze_traffic_flow(tracked_objects)

        # Assess situation complexity
        situation_complexity = self._assess_complexity(
            object_count, moving_count, speed_variance, behaviors
        )

        # Count behavioral anomalies
        erratic_count = self._count_erratic_behaviors(behaviors) if behaviors else 0

        # Count collision risks
        collision_risk_count = sum(
            1 for obj in tracked_objects
            if obj.time_to_collision is not None and obj.time_to_collision < 3.0
        )

        # Assess danger level
        danger_level = self._assess_danger_level(
            traffic_density, collision_risk_count, erratic_count, situation_complexity
        )

        # Check for special situations
        is_intersection = self._detect_intersection(traffic_flow, behaviors)
        is_crowded = spatial_density > 0.5 or object_count > 10
        has_pedestrians = self._detect_pedestrians(tracked_objects)
        has_vulnerable_users = has_pedestrians or self._detect_cyclists(tracked_objects)

        # Lane keeping quality
        lane_keeping = self._assess_lane_keeping(lane_result) if lane_result else None

        return SceneAnalysis(
            timestamp=timestamp,
            traffic_density=traffic_density,
            scene_type=scene_type,
            traffic_flow=traffic_flow,
            situation_complexity=situation_complexity,
            danger_level=danger_level,
            object_count=object_count,
            moving_object_count=moving_count,
            average_speed=avg_speed,
            speed_variance=speed_variance,
            spatial_density=spatial_density,
            erratic_behavior_count=erratic_count,
            collision_risk_count=collision_risk_count,
            lane_keeping_quality=lane_keeping,
            is_intersection=is_intersection,
            is_crowded=is_crowded,
            has_pedestrians=has_pedestrians,
            has_vulnerable_road_users=has_vulnerable_users
        )

    def _calculate_spatial_density(
        self,
        tracked_objects: List[TrackedObject]
    ) -> float:
        """Calculate spatial density (objects per square meter)."""
        if not tracked_objects:
            return 0.0

        # Count objects within spatial range
        in_range_count = 0
        for obj in tracked_objects:
            if obj.current_position:
                x, y, z = obj.current_position
                distance = np.sqrt(x**2 + y**2)
                if distance <= self.spatial_range:
                    in_range_count += 1

        # Calculate density
        area = np.pi * self.spatial_range**2  # Circular area
        density = in_range_count / area

        return density

    def _classify_traffic_density(
        self,
        object_count: int,
        spatial_density: float
    ) -> TrafficDensity:
        """Classify traffic density level."""
        if object_count == 0:
            return TrafficDensity.EMPTY

        if object_count <= 2 and spatial_density < 0.01:
            return TrafficDensity.LIGHT

        if object_count <= 5 and spatial_density < 0.05:
            return TrafficDensity.MODERATE

        if object_count <= 10 and spatial_density < 0.1:
            return TrafficDensity.HEAVY

        return TrafficDensity.CONGESTED

    def _classify_scene_type(
        self,
        tracked_objects: List[TrackedObject],
        avg_speed: float,
        spatial_density: float
    ) -> SceneType:
        """Classify the scene type."""
        object_count = len(tracked_objects)

        # Highway: High speed, low density
        if avg_speed > 15.0 and spatial_density < 0.05:  # ~55 km/h+
            return SceneType.HIGHWAY

        # Parking lot: Low speed, possibly high density, many stationary
        stationary_count = sum(1 for obj in tracked_objects if obj.is_stationary)
        if avg_speed < 2.0 and stationary_count > object_count / 2:
            return SceneType.PARKING_LOT

        # Urban street: Moderate speed, moderate density
        if 5.0 < avg_speed < 15.0:
            return SceneType.URBAN_STREET

        # Residential: Low-moderate speed, low density
        if avg_speed < 8.0 and spatial_density < 0.03:
            return SceneType.RESIDENTIAL

        return SceneType.UNKNOWN

    def _analyze_traffic_flow(
        self,
        tracked_objects: List[TrackedObject]
    ) -> TrafficFlow:
        """Analyze traffic flow patterns."""
        if not tracked_objects:
            return TrafficFlow.STATIC

        # Get velocity directions
        directions = []
        for obj in tracked_objects:
            if obj.current_velocity and not obj.is_stationary:
                vx, vy = obj.current_velocity
                angle = np.arctan2(vy, vx)
                directions.append(angle)

        if not directions:
            return TrafficFlow.STATIC

        if len(directions) == 1:
            return TrafficFlow.SAME_DIRECTION

        # Calculate angular variance
        directions = np.array(directions)
        mean_direction = np.arctan2(
            np.mean(np.sin(directions)),
            np.mean(np.cos(directions))
        )

        # Calculate angular deviation
        deviations = []
        for angle in directions:
            diff = angle - mean_direction
            # Normalize to [-π, π]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            deviations.append(abs(diff))

        avg_deviation = np.mean(deviations)

        # Classify flow
        if avg_deviation < np.pi / 6:  # ~30 degrees
            return TrafficFlow.SAME_DIRECTION
        elif avg_deviation < np.pi / 3:  # ~60 degrees
            return TrafficFlow.OPPOSITE_DIRECTIONS
        else:
            return TrafficFlow.MULTI_DIRECTIONAL

    def _assess_complexity(
        self,
        object_count: int,
        moving_count: int,
        speed_variance: float,
        behaviors: Optional[Dict[int, ObjectBehavior]]
    ) -> SituationComplexity:
        """Assess situational complexity."""
        complexity_score = 0

        # Object count contribution
        if object_count > 10:
            complexity_score += 3
        elif object_count > 5:
            complexity_score += 2
        elif object_count > 2:
            complexity_score += 1

        # Movement contribution
        if moving_count > 7:
            complexity_score += 2
        elif moving_count > 3:
            complexity_score += 1

        # Speed variance contribution (indicates unpredictable behavior)
        if speed_variance > 10.0:
            complexity_score += 2
        elif speed_variance > 5.0:
            complexity_score += 1

        # Behavioral complexity
        if behaviors:
            maneuver_count = sum(
                1 for b in behaviors.values()
                if b.maneuver_type.value not in ["normal_driving", "unknown"]
            )
            if maneuver_count > 3:
                complexity_score += 2
            elif maneuver_count > 1:
                complexity_score += 1

        # Classify based on score
        if complexity_score <= 2:
            return SituationComplexity.SIMPLE
        elif complexity_score <= 5:
            return SituationComplexity.MODERATE
        elif complexity_score <= 8:
            return SituationComplexity.COMPLEX
        else:
            return SituationComplexity.CRITICAL

    def _count_erratic_behaviors(
        self,
        behaviors: Dict[int, ObjectBehavior]
    ) -> int:
        """Count objects with erratic or unusual behaviors."""
        erratic_count = 0

        for behavior in behaviors.values():
            # Consider these behaviors as potentially erratic
            if behavior.maneuver_type.value in ["u_turn", "reversing"]:
                erratic_count += 1
            # Sharp turns
            elif behavior.turn_rate and abs(behavior.turn_rate) > 20.0:
                erratic_count += 1
            # Hard braking
            elif behavior.acceleration and behavior.acceleration < -3.0:
                erratic_count += 1

        return erratic_count

    def _assess_danger_level(
        self,
        traffic_density: TrafficDensity,
        collision_risk_count: int,
        erratic_count: int,
        complexity: SituationComplexity
    ) -> DangerLevel:
        """Assess overall danger level."""
        danger_score = 0

        # Collision risks
        if collision_risk_count > 3:
            danger_score += 4
        elif collision_risk_count > 1:
            danger_score += 2
        elif collision_risk_count > 0:
            danger_score += 1

        # Erratic behaviors
        if erratic_count > 2:
            danger_score += 3
        elif erratic_count > 0:
            danger_score += 1

        # Complexity
        if complexity == SituationComplexity.CRITICAL:
            danger_score += 3
        elif complexity == SituationComplexity.COMPLEX:
            danger_score += 2
        elif complexity == SituationComplexity.MODERATE:
            danger_score += 1

        # Traffic density (congestion can be dangerous)
        if traffic_density == TrafficDensity.CONGESTED:
            danger_score += 2

        # Classify danger
        if danger_score == 0:
            return DangerLevel.SAFE
        elif danger_score <= 3:
            return DangerLevel.CAUTIOUS
        elif danger_score <= 6:
            return DangerLevel.WARNING
        else:
            return DangerLevel.DANGER

    def _detect_intersection(
        self,
        traffic_flow: TrafficFlow,
        behaviors: Optional[Dict[int, ObjectBehavior]]
    ) -> bool:
        """Detect if scene appears to be an intersection."""
        # Multi-directional flow suggests intersection
        if traffic_flow == TrafficFlow.MULTI_DIRECTIONAL:
            return True

        # Multiple turning objects suggests intersection
        if behaviors:
            turning_count = sum(
                1 for b in behaviors.values()
                if b.turning_behavior.value in ["turning_left", "turning_right"]
            )
            if turning_count > 2:
                return True

        return False

    def _detect_pedestrians(self, tracked_objects: List[TrackedObject]) -> bool:
        """Detect if pedestrians are present."""
        for obj in tracked_objects:
            if obj.class_name.lower() == "person":
                return True
        return False

    def _detect_cyclists(self, tracked_objects: List[TrackedObject]) -> bool:
        """Detect if cyclists are present."""
        for obj in tracked_objects:
            if obj.class_name.lower() in ["bicycle", "motorcycle"]:
                return True
        return False

    def _assess_lane_keeping(self, lane_result: LaneDetectionResult) -> float:
        """Assess lane keeping quality (0-1)."""
        # Based on lateral offset and detection confidence
        if lane_result.lateral_offset is None:
            return 0.5  # Unknown

        offset = abs(lane_result.lateral_offset)

        # Good lane keeping: within 0.5m of center
        if offset < 0.5:
            quality = 1.0
        elif offset < 1.0:
            quality = 0.7
        elif offset < 1.5:
            quality = 0.4
        else:
            quality = 0.2

        # Adjust by detection confidence
        quality *= lane_result.detection_confidence

        return quality

    def get_recent_trend(self, metric: str) -> Optional[str]:
        """
        Get trend for a specific metric (improving/worsening/stable).

        Args:
            metric: Metric name (e.g., "danger_level", "traffic_density")

        Returns:
            "improving", "worsening", "stable", or None
        """
        if len(self.analysis_history) < 3:
            return None

        recent = list(self.analysis_history)[-5:]

        try:
            # Get metric values
            values = [getattr(analysis, metric) for analysis in recent]

            # For enum types, compare ordinal values
            if hasattr(values[0], 'value'):
                # Convert to enum indices for comparison
                enum_class = type(values[0])
                indices = [list(enum_class).index(v) for v in values]

                # Check trend
                if indices[-1] > indices[0]:
                    return "worsening"
                elif indices[-1] < indices[0]:
                    return "improving"
                else:
                    return "stable"

        except Exception as e:
            logger.debug(f"Trend calculation failed for {metric}: {e}")

        return None
