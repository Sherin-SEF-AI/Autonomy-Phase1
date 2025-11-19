"""
Predictive Collision Warning System

This module provides advanced predictive collision detection and warning,
including:
- Multi-object trajectory prediction
- Collision probability calculation
- Time-to-collision (TTC) estimation
- Critical zone monitoring
- Risk assessment and prioritization
- Multiple collision scenarios (frontal, rear, side, pedestrian)
- Early warning generation
- Recommended actions

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math
import time


class CollisionType(Enum):
    """Types of potential collisions"""
    FRONTAL = "frontal"
    REAR = "rear"
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    INTERSECTION = "intersection"
    LANE_CHANGE = "lane_change"
    BACKING = "backing"


class RiskLevel(Enum):
    """Collision risk levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    IMMINENT = "imminent"


class RecommendedAction(Enum):
    """Recommended driver actions"""
    NONE = "none"
    MONITOR = "monitor"
    PREPARE_BRAKE = "prepare_brake"
    BRAKE_GENTLY = "brake_gently"
    BRAKE_HARD = "brake_hard"
    EMERGENCY_BRAKE = "emergency_brake"
    STEER_LEFT = "steer_left"
    STEER_RIGHT = "steer_right"
    STOP = "stop"


@dataclass
class PredictedObject:
    """Object with predicted future trajectory"""
    object_id: int
    current_position: Tuple[float, float, float]  # x, y, z
    current_velocity: Tuple[float, float, float]  # vx, vy, vz
    current_acceleration: Tuple[float, float, float]  # ax, ay, az
    predicted_positions: List[Tuple[float, float, float]]  # Future positions
    predicted_times: List[float]  # Time stamps for predictions
    class_name: str
    bbox_size: Tuple[float, float, float]  # width, height, depth
    confidence: float


@dataclass
class CollisionWarning:
    """Collision warning information"""
    warning_id: int
    collision_type: CollisionType
    risk_level: RiskLevel
    time_to_collision: float  # seconds
    collision_probability: float  # 0-1
    collision_point: Optional[Tuple[float, float, float]]  # Predicted collision location
    involved_objects: List[int]  # Object IDs
    recommended_action: RecommendedAction
    message: str
    timestamp: float


@dataclass
class CriticalZone:
    """Critical monitoring zone around vehicle"""
    zone_id: str
    polygon: List[Tuple[float, float]]  # Zone boundary points
    risk_weight: float  # Importance weight for this zone
    monitored_object_types: List[str]  # Object classes to monitor


@dataclass
class PCWConfig:
    """Configuration for Predictive Collision Warning"""
    # Prediction horizon
    prediction_time: float = 5.0  # seconds ahead
    prediction_steps: int = 10  # Number of prediction steps

    # TTC thresholds for different risk levels
    ttc_critical: float = 1.0  # seconds
    ttc_high: float = 2.0
    ttc_medium: float = 3.5
    ttc_low: float = 5.0

    # Collision probability thresholds
    prob_imminent: float = 0.9
    prob_critical: float = 0.7
    prob_high: float = 0.5
    prob_medium: float = 0.3

    # Safety margins
    longitudinal_margin: float = 2.0  # meters
    lateral_margin: float = 1.0  # meters
    vertical_margin: float = 0.5  # meters

    # Vehicle dimensions (for collision volume calculation)
    vehicle_length: float = 4.5
    vehicle_width: float = 1.8
    vehicle_height: float = 1.5

    # Special object handling
    pedestrian_safety_margin: float = 3.0  # Extra margin for pedestrians
    cyclist_safety_margin: float = 2.5  # Extra margin for cyclists
    vulnerable_road_user_priority: float = 2.0  # Priority multiplier

    # Warning filtering
    min_probability_threshold: float = 0.2
    min_warning_duration: float = 0.5  # seconds


class PredictiveCollisionWarning:
    """
    Predictive Collision Warning System

    Features:
    - Multi-object trajectory prediction
    - Advanced TTC calculation
    - Collision probability estimation
    - Critical zone monitoring
    - Risk level classification
    - Action recommendations
    - Multi-scenario collision detection
    - Vulnerable road user prioritization
    """

    def __init__(self, config: Optional[PCWConfig] = None):
        """
        Initialize Predictive Collision Warning system

        Args:
            config: PCW configuration
        """
        self.config = config or PCWConfig()
        self.active_warnings: Dict[int, CollisionWarning] = {}
        self.next_warning_id = 0
        self.warning_history: List[CollisionWarning] = []

        # Define critical zones around vehicle
        self.critical_zones = self._define_critical_zones()

        # Statistics
        self.total_warnings_generated = 0
        self.warnings_by_type: Dict[CollisionType, int] = {}
        self.false_positives = 0  # Would need ground truth to calculate

    def _define_critical_zones(self) -> List[CriticalZone]:
        """Define critical monitoring zones around vehicle"""
        vl = self.config.vehicle_length
        vw = self.config.vehicle_width

        zones = [
            # Front zone (most critical)
            CriticalZone(
                zone_id="front",
                polygon=[
                    (-vw/2, 0), (vw/2, 0),
                    (vw/2, vl * 2), (-vw/2, vl * 2)
                ],
                risk_weight=1.0,
                monitored_object_types=["car", "truck", "bus", "person", "bicycle", "motorcycle"]
            ),
            # Rear zone
            CriticalZone(
                zone_id="rear",
                polygon=[
                    (-vw/2, -vl), (vw/2, -vl),
                    (vw/2, 0), (-vw/2, 0)
                ],
                risk_weight=0.6,
                monitored_object_types=["car", "truck", "bus"]
            ),
            # Left side zone
            CriticalZone(
                zone_id="left",
                polygon=[
                    (-vw/2 - vw, -vl/2),
                    (-vw/2, -vl/2),
                    (-vw/2, vl/2),
                    (-vw/2 - vw, vl/2)
                ],
                risk_weight=0.7,
                monitored_object_types=["car", "truck", "bus", "motorcycle", "bicycle"]
            ),
            # Right side zone
            CriticalZone(
                zone_id="right",
                polygon=[
                    (vw/2, -vl/2),
                    (vw/2 + vw, -vl/2),
                    (vw/2 + vw, vl/2),
                    (vw/2, vl/2)
                ],
                risk_weight=0.7,
                monitored_object_types=["car", "truck", "bus", "motorcycle", "bicycle"]
            ),
        ]

        return zones

    def predict_and_warn(
        self,
        tracked_objects: List[Dict[str, Any]],
        ego_state: Dict[str, Any],
        timestamp: Optional[float] = None
    ) -> List[CollisionWarning]:
        """
        Predict future states and generate collision warnings

        Args:
            tracked_objects: List of tracked objects with positions, velocities
            ego_state: Ego vehicle state (position, velocity, acceleration)
            timestamp: Current timestamp

        Returns:
            List of active collision warnings
        """
        if timestamp is None:
            timestamp = time.time()

        # Predict future trajectories for all objects
        predicted_objects = []
        for obj in tracked_objects:
            pred_obj = self._predict_object_trajectory(obj)
            if pred_obj:
                predicted_objects.append(pred_obj)

        # Predict ego vehicle trajectory
        ego_predicted = self._predict_ego_trajectory(ego_state)

        # Detect potential collisions
        new_warnings = []
        for pred_obj in predicted_objects:
            warning = self._check_collision(ego_predicted, pred_obj, timestamp)
            if warning:
                new_warnings.append(warning)

        # Update active warnings
        self._update_warnings(new_warnings, timestamp)

        # Get current active warnings
        active = [w for w in self.active_warnings.values()]

        # Sort by risk level and TTC
        active.sort(key=lambda w: (
            self._risk_level_priority(w.risk_level),
            w.time_to_collision
        ))

        return active

    def _predict_object_trajectory(
        self,
        obj: Dict[str, Any]
    ) -> Optional[PredictedObject]:
        """Predict future trajectory of an object"""
        position = obj.get('position')
        velocity = obj.get('velocity')
        acceleration = obj.get('acceleration', (0, 0, 0))

        if position is None or velocity is None:
            return None

        # Ensure 3D coordinates
        if len(position) == 2:
            position = (*position, 0)
        if len(velocity) == 2:
            velocity = (*velocity, 0)
        if len(acceleration) == 2:
            acceleration = (*acceleration, 0)

        # Predict future positions using kinematic equations
        predicted_positions = []
        predicted_times = []

        dt = self.config.prediction_time / self.config.prediction_steps

        for i in range(self.config.prediction_steps):
            t = (i + 1) * dt

            # s = s0 + v0*t + 0.5*a*t^2
            x = position[0] + velocity[0] * t + 0.5 * acceleration[0] * t**2
            y = position[1] + velocity[1] * t + 0.5 * acceleration[1] * t**2
            z = position[2] + velocity[2] * t + 0.5 * acceleration[2] * t**2

            predicted_positions.append((x, y, z))
            predicted_times.append(t)

        return PredictedObject(
            object_id=obj.get('id', -1),
            current_position=position,
            current_velocity=velocity,
            current_acceleration=acceleration,
            predicted_positions=predicted_positions,
            predicted_times=predicted_times,
            class_name=obj.get('class', 'unknown'),
            bbox_size=obj.get('size', (2.0, 2.0, 2.0)),
            confidence=obj.get('confidence', 1.0)
        )

    def _predict_ego_trajectory(
        self,
        ego_state: Dict[str, Any]
    ) -> PredictedObject:
        """Predict ego vehicle future trajectory"""
        position = ego_state.get('position', (0, 0, 0))
        velocity = ego_state.get('velocity', (0, 0, 0))
        acceleration = ego_state.get('acceleration', (0, 0, 0))

        # Similar prediction as objects
        predicted_positions = []
        predicted_times = []

        dt = self.config.prediction_time / self.config.prediction_steps

        for i in range(self.config.prediction_steps):
            t = (i + 1) * dt

            x = position[0] + velocity[0] * t + 0.5 * acceleration[0] * t**2
            y = position[1] + velocity[1] * t + 0.5 * acceleration[1] * t**2
            z = position[2] + velocity[2] * t + 0.5 * acceleration[2] * t**2

            predicted_positions.append((x, y, z))
            predicted_times.append(t)

        return PredictedObject(
            object_id=-1,  # Ego vehicle
            current_position=position,
            current_velocity=velocity,
            current_acceleration=acceleration,
            predicted_positions=predicted_positions,
            predicted_times=predicted_times,
            class_name="ego",
            bbox_size=(self.config.vehicle_width,
                      self.config.vehicle_height,
                      self.config.vehicle_length),
            confidence=1.0
        )

    def _check_collision(
        self,
        ego: PredictedObject,
        obj: PredictedObject,
        timestamp: float
    ) -> Optional[CollisionWarning]:
        """Check for potential collision between ego and object"""
        # Find closest approach
        min_distance = float('inf')
        collision_time = None
        collision_point = None
        collision_step = -1

        for i, (ego_pos, obj_pos, t) in enumerate(zip(
            ego.predicted_positions,
            obj.predicted_positions,
            ego.predicted_times
        )):
            distance = math.sqrt(
                (ego_pos[0] - obj_pos[0])**2 +
                (ego_pos[1] - obj_pos[1])**2 +
                (ego_pos[2] - obj_pos[2])**2
            )

            if distance < min_distance:
                min_distance = distance
                collision_time = t
                collision_point = (
                    (ego_pos[0] + obj_pos[0]) / 2,
                    (ego_pos[1] + obj_pos[1]) / 2,
                    (ego_pos[2] + obj_pos[2]) / 2
                )
                collision_step = i

        # Determine if collision is likely
        # Sum of radii (approximate objects as spheres for simplicity)
        ego_radius = max(ego.bbox_size) / 2
        obj_radius = max(obj.bbox_size) / 2

        # Add safety margins
        safety_margin = self.config.longitudinal_margin
        if obj.class_name == "person":
            safety_margin += self.config.pedestrian_safety_margin
        elif obj.class_name == "bicycle":
            safety_margin += self.config.cyclist_safety_margin

        threshold_distance = ego_radius + obj_radius + safety_margin

        if min_distance > threshold_distance:
            # No collision risk
            return None

        # Calculate collision probability
        # Based on distance, relative velocity, and prediction confidence
        prob = self._calculate_collision_probability(
            min_distance, threshold_distance, obj.confidence
        )

        if prob < self.config.min_probability_threshold:
            return None

        # Classify collision type
        collision_type = self._classify_collision_type(
            ego.current_position, ego.current_velocity,
            obj.current_position, obj.current_velocity,
            obj.class_name
        )

        # Determine risk level
        risk_level = self._determine_risk_level(collision_time, prob)

        # Recommend action
        recommended_action = self._recommend_action(risk_level, collision_type, collision_time)

        # Generate warning message
        message = self._generate_warning_message(
            collision_type, risk_level, collision_time, obj.class_name
        )

        # Create warning
        warning = CollisionWarning(
            warning_id=self.next_warning_id,
            collision_type=collision_type,
            risk_level=risk_level,
            time_to_collision=collision_time if collision_time else 0.0,
            collision_probability=prob,
            collision_point=collision_point,
            involved_objects=[obj.object_id],
            recommended_action=recommended_action,
            message=message,
            timestamp=timestamp
        )

        self.next_warning_id += 1
        self.total_warnings_generated += 1
        self.warnings_by_type[collision_type] = \
            self.warnings_by_type.get(collision_type, 0) + 1

        return warning

    def _calculate_collision_probability(
        self,
        distance: float,
        threshold: float,
        confidence: float
    ) -> float:
        """Calculate probability of collision"""
        # Sigmoid-based probability
        # As distance approaches threshold, probability increases
        ratio = distance / threshold if threshold > 0 else 0

        # Probability decreases with distance
        if ratio >= 1.5:
            prob = 0.0
        elif ratio >= 1.0:
            prob = 1.0 - (ratio - 1.0) / 0.5
        else:
            prob = 1.0 - ratio * 0.5

        # Modulate by object detection confidence
        prob *= confidence

        return np.clip(prob, 0.0, 1.0)

    def _classify_collision_type(
        self,
        ego_pos: Tuple[float, float, float],
        ego_vel: Tuple[float, float, float],
        obj_pos: Tuple[float, float, float],
        obj_vel: Tuple[float, float, float],
        obj_class: str
    ) -> CollisionType:
        """Classify type of collision scenario"""
        # Relative position
        dx = obj_pos[0] - ego_pos[0]
        dy = obj_pos[1] - ego_pos[1]

        # Special cases for vulnerable road users
        if obj_class == "person":
            return CollisionType.PEDESTRIAN
        elif obj_class in ["bicycle", "motorcycle"]:
            return CollisionType.CYCLIST

        # Determine direction
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle) % 360

        # Frontal collision (object ahead)
        if 315 <= angle_deg or angle_deg < 45:
            return CollisionType.FRONTAL
        # Side collisions
        elif 45 <= angle_deg < 135:
            return CollisionType.SIDE_RIGHT
        elif 225 <= angle_deg < 315:
            return CollisionType.SIDE_LEFT
        # Rear collision
        elif 135 <= angle_deg < 225:
            return CollisionType.REAR

        return CollisionType.FRONTAL

    def _determine_risk_level(self, ttc: float, probability: float) -> RiskLevel:
        """Determine risk level from TTC and probability"""
        if probability >= self.config.prob_imminent or ttc <= self.config.ttc_critical:
            return RiskLevel.IMMINENT
        elif probability >= self.config.prob_critical or ttc <= self.config.ttc_high:
            return RiskLevel.CRITICAL
        elif probability >= self.config.prob_high or ttc <= self.config.ttc_medium:
            return RiskLevel.HIGH
        elif probability >= self.config.prob_medium or ttc <= self.config.ttc_low:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _recommend_action(
        self,
        risk_level: RiskLevel,
        collision_type: CollisionType,
        ttc: float
    ) -> RecommendedAction:
        """Recommend action based on risk and scenario"""
        if risk_level == RiskLevel.IMMINENT:
            if collision_type == CollisionType.FRONTAL:
                return RecommendedAction.EMERGENCY_BRAKE
            elif collision_type == CollisionType.SIDE_LEFT:
                return RecommendedAction.STEER_RIGHT
            elif collision_type == CollisionType.SIDE_RIGHT:
                return RecommendedAction.STEER_LEFT
            else:
                return RecommendedAction.EMERGENCY_BRAKE

        elif risk_level == RiskLevel.CRITICAL:
            return RecommendedAction.BRAKE_HARD

        elif risk_level == RiskLevel.HIGH:
            return RecommendedAction.BRAKE_GENTLY

        elif risk_level == RiskLevel.MEDIUM:
            return RecommendedAction.PREPARE_BRAKE

        elif risk_level == RiskLevel.LOW:
            return RecommendedAction.MONITOR

        return RecommendedAction.NONE

    def _generate_warning_message(
        self,
        collision_type: CollisionType,
        risk_level: RiskLevel,
        ttc: float,
        obj_class: str
    ) -> str:
        """Generate human-readable warning message"""
        if risk_level == RiskLevel.IMMINENT:
            return f"COLLISION IMMINENT! {obj_class.upper()} ahead in {ttc:.1f}s - BRAKE NOW!"
        elif risk_level == RiskLevel.CRITICAL:
            return f"CRITICAL: {obj_class.capitalize()} collision risk - {ttc:.1f}s - Brake immediately!"
        elif risk_level == RiskLevel.HIGH:
            return f"WARNING: Potential {collision_type.value} collision with {obj_class} in {ttc:.1f}s"
        elif risk_level == RiskLevel.MEDIUM:
            return f"CAUTION: {obj_class.capitalize()} ahead - prepare to brake"
        else:
            return f"Monitor {obj_class} ahead"

    def _risk_level_priority(self, risk: RiskLevel) -> int:
        """Get priority value for risk level (lower = higher priority)"""
        priority_map = {
            RiskLevel.IMMINENT: 0,
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4,
            RiskLevel.NONE: 5
        }
        return priority_map.get(risk, 10)

    def _update_warnings(
        self,
        new_warnings: List[CollisionWarning],
        timestamp: float
    ):
        """Update active warnings with new detections"""
        # Remove stale warnings
        to_remove = []
        for wid, warning in self.active_warnings.items():
            if timestamp - warning.timestamp > self.config.min_warning_duration * 2:
                to_remove.append(wid)

        for wid in to_remove:
            del self.active_warnings[wid]

        # Add new warnings
        for warning in new_warnings:
            self.active_warnings[warning.warning_id] = warning
            self.warning_history.append(warning)

    def visualize_warnings(
        self,
        image: np.ndarray,
        warnings: List[CollisionWarning]
    ) -> np.ndarray:
        """Visualize collision warnings on image"""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        if not warnings:
            return vis_image

        # Draw warning panel
        panel_height = min(200, len(warnings) * 50 + 50)
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)

        # Most critical warning
        top_warning = warnings[0]

        # Background color based on risk
        bg_color = self._get_risk_color(top_warning.risk_level)
        panel[:] = bg_color

        # Draw warnings
        y_offset = 30
        for i, warning in enumerate(warnings[:4]):  # Show top 4
            color = (255, 255, 255)
            text = f"{warning.message}"

            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # TTC and probability
            info = f"TTC: {warning.time_to_collision:.1f}s | Prob: {warning.collision_probability:.0%}"
            cv2.putText(panel, info, (10, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            y_offset += 50

        # Combine with image
        result = np.vstack([vis_image, panel])

        return result

    def _get_risk_color(self, risk: RiskLevel) -> Tuple[int, int, int]:
        """Get BGR color for risk level"""
        color_map = {
            RiskLevel.IMMINENT: (0, 0, 128),      # Dark red
            RiskLevel.CRITICAL: (0, 0, 255),      # Red
            RiskLevel.HIGH: (0, 100, 255),        # Orange
            RiskLevel.MEDIUM: (0, 255, 255),      # Yellow
            RiskLevel.LOW: (0, 255, 0),           # Green
            RiskLevel.NONE: (50, 50, 50)          # Gray
        }
        return color_map.get(risk, (0, 0, 0))

    def get_statistics(self) -> Dict[str, Any]:
        """Get PCW statistics"""
        return {
            'total_warnings': self.total_warnings_generated,
            'active_warnings': len(self.active_warnings),
            'warnings_by_type': {k.value: v for k, v in self.warnings_by_type.items()},
            'warning_history_size': len(self.warning_history)
        }
