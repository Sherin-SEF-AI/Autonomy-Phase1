"""
Real-time safety scoring system.

Evaluates driving safety based on:
- Collision risks
- Following distance
- Lane keeping
- Speed appropriateness
- Environmental conditions
- Dangerous maneuvers
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from utils.data_structures import TrackedObject, LaneDetectionResult, SafetyWarning, WarningLevel
from perception.behavior_classification import ObjectBehavior
from perception.scene_understanding import SceneAnalysis
from perception.scene_recognition import SceneContext
from utils.logger import get_logger


logger = get_logger()


class SafetyLevel(Enum):
    """Overall safety levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"  # 75-89
    FAIR = "fair"  # 60-74
    POOR = "poor"  # 40-59
    CRITICAL = "critical"  # 0-39


@dataclass
class SafetyEvent:
    """Represents a safety event."""
    timestamp: float
    event_type: str
    severity: str
    score_impact: float  # Negative value
    description: str


@dataclass
class SafetyScore:
    """Complete safety assessment."""
    timestamp: float

    # Overall score (0-100)
    overall_score: float
    safety_level: SafetyLevel

    # Component scores (0-100 each)
    collision_avoidance_score: float
    lane_keeping_score: float
    following_distance_score: float
    speed_appropriateness_score: float
    environmental_awareness_score: float

    # Event counters
    near_misses_count: int = 0
    hard_braking_count: int = 0
    lane_departures_count: int = 0
    dangerous_maneuvers_count: int = 0

    # Recent events
    recent_events: List[SafetyEvent] = field(default_factory=list)


class SafetyScorer:
    """
    Calculates real-time safety scores based on driving behavior and environment.
    """

    def __init__(
        self,
        event_history_duration: float = 300.0,  # 5 minutes
        score_history_length: int = 100
    ):
        """
        Initialize safety scorer.

        Args:
            event_history_duration: How long to keep events in history (seconds)
            score_history_length: Number of scores to keep for trending
        """
        self.event_history_duration = event_history_duration
        self.score_history_length = score_history_length

        # Event history
        self.safety_events: deque = deque()

        # Score history
        self.score_history: deque = deque(maxlen=score_history_length)

        # Cumulative counters
        self.total_near_misses = 0
        self.total_hard_braking = 0
        self.total_lane_departures = 0
        self.total_dangerous_maneuvers = 0

        # Session tracking
        self.session_start_time = time.time()
        self.total_distance_km = 0.0  # Would need GPS/odometry

        logger.info("Safety scorer initialized")

    def calculate_score(
        self,
        tracked_objects: List[TrackedObject],
        lane_result: Optional[LaneDetectionResult],
        scene_analysis: Optional[SceneAnalysis],
        scene_context: Optional[SceneContext],
        warnings: List[SafetyWarning],
        behaviors: Optional[Dict[int, ObjectBehavior]] = None,
        ego_speed: Optional[float] = None  # m/s
    ) -> SafetyScore:
        """
        Calculate comprehensive safety score.

        Args:
            tracked_objects: Currently tracked objects
            lane_result: Lane detection result
            scene_analysis: Scene understanding analysis
            scene_context: Scene recognition context
            warnings: Active safety warnings
            behaviors: Object behaviors
            ego_speed: Ego vehicle speed in m/s

        Returns:
            SafetyScore
        """
        current_time = time.time()

        # Calculate component scores
        collision_score = self._score_collision_avoidance(
            tracked_objects, warnings
        )

        lane_score = self._score_lane_keeping(lane_result)

        following_score = self._score_following_distance(
            tracked_objects, ego_speed
        )

        speed_score = self._score_speed_appropriateness(
            ego_speed, scene_context, scene_analysis
        )

        env_score = self._score_environmental_awareness(
            scene_context, scene_analysis
        )

        # Calculate weighted overall score
        weights = {
            'collision': 0.35,  # Most important
            'lane': 0.20,
            'following': 0.20,
            'speed': 0.15,
            'environment': 0.10
        }

        overall_score = (
            collision_score * weights['collision'] +
            lane_score * weights['lane'] +
            following_score * weights['following'] +
            speed_score * weights['speed'] +
            env_score * weights['environment']
        )

        # Determine safety level
        safety_level = self._get_safety_level(overall_score)

        # Get recent events
        recent_events = self._get_recent_events(current_time)

        # Count events in current window
        near_misses = sum(
            1 for e in recent_events
            if e.event_type == "near_miss"
        )
        hard_braking = sum(
            1 for e in recent_events
            if e.event_type == "hard_braking"
        )
        lane_departures = sum(
            1 for e in recent_events
            if e.event_type == "lane_departure"
        )
        dangerous_maneuvers = sum(
            1 for e in recent_events
            if e.event_type == "dangerous_maneuver"
        )

        # Create score object
        score = SafetyScore(
            timestamp=current_time,
            overall_score=overall_score,
            safety_level=safety_level,
            collision_avoidance_score=collision_score,
            lane_keeping_score=lane_score,
            following_distance_score=following_score,
            speed_appropriateness_score=speed_score,
            environmental_awareness_score=env_score,
            near_misses_count=near_misses,
            hard_braking_count=hard_braking,
            lane_departures_count=lane_departures,
            dangerous_maneuvers_count=dangerous_maneuvers,
            recent_events=recent_events[-5:]  # Last 5 events
        )

        # Store in history
        self.score_history.append(score)

        return score

    def _score_collision_avoidance(
        self,
        tracked_objects: List[TrackedObject],
        warnings: List[SafetyWarning]
    ) -> float:
        """Score collision avoidance (0-100)."""
        score = 100.0

        # Penalize based on collision warnings
        for warning in warnings:
            if warning.warning_type.value in ["forward_collision", "pedestrian_collision"]:
                if warning.level == WarningLevel.CRITICAL:
                    score -= 30
                    self._log_event("near_miss", "critical", -30, "Critical collision warning")
                elif warning.level == WarningLevel.CAUTION:
                    score -= 15
                elif warning.level == WarningLevel.ADVISORY:
                    score -= 5

        # Penalize for close objects with low TTC
        for obj in tracked_objects:
            if obj.time_to_collision is not None:
                if obj.time_to_collision < 1.0:
                    score -= 20
                elif obj.time_to_collision < 2.0:
                    score -= 10
                elif obj.time_to_collision < 3.0:
                    score -= 5

        return max(0.0, min(100.0, score))

    def _score_lane_keeping(self, lane_result: Optional[LaneDetectionResult]) -> float:
        """Score lane keeping (0-100)."""
        if not lane_result:
            return 75.0  # Neutral if no lane data

        score = 100.0

        # Check for lane departure
        if lane_result.departure_warning:
            score -= 25
            self._log_event(
                "lane_departure",
                "high",
                -25,
                f"Lane departure {lane_result.departure_direction}"
            )

        # Penalize for lateral offset
        if lane_result.lateral_offset is not None:
            offset = abs(lane_result.lateral_offset)
            if offset > 1.0:  # More than 1m from center
                score -= 20
            elif offset > 0.7:
                score -= 10
            elif offset > 0.5:
                score -= 5

        # Reward good detection confidence
        if lane_result.detection_confidence > 0.8:
            score += 5  # Bonus for good lane tracking

        return max(0.0, min(100.0, score))

    def _score_following_distance(
        self,
        tracked_objects: List[TrackedObject],
        ego_speed: Optional[float]
    ) -> float:
        """Score following distance (0-100)."""
        if ego_speed is None or ego_speed < 1.0:  # Stationary
            return 100.0

        # Find lead vehicle (object directly ahead)
        lead_vehicle = None
        min_distance = float('inf')

        for obj in tracked_objects:
            if obj.class_name.lower() in ["car", "truck", "bus"]:
                if obj.current_position:
                    x, y, z = obj.current_position
                    # Check if ahead (positive x) and close to center (small y)
                    if x > 0 and abs(y) < 2.0:  # Within 2m of centerline
                        distance = np.sqrt(x**2 + y**2)
                        if distance < min_distance:
                            min_distance = distance
                            lead_vehicle = obj

        if lead_vehicle is None:
            return 100.0  # No lead vehicle

        # Calculate safe following distance (2-second rule)
        safe_distance = ego_speed * 2.0  # meters

        score = 100.0

        if min_distance < safe_distance * 0.5:  # Less than half safe distance
            score = 30.0
            self._log_event("following_too_close", "critical", -70, "Following too close")
        elif min_distance < safe_distance * 0.75:
            score = 60.0
        elif min_distance < safe_distance:
            score = 80.0
        else:
            score = 100.0

        return score

    def _score_speed_appropriateness(
        self,
        ego_speed: Optional[float],
        scene_context: Optional[SceneContext],
        scene_analysis: Optional[SceneAnalysis]
    ) -> float:
        """Score speed appropriateness for conditions (0-100)."""
        if ego_speed is None:
            return 75.0  # Neutral

        score = 100.0

        # Adjust for visibility
        if scene_context and scene_context.visibility_score < 0.5:
            # Poor visibility - should reduce speed
            if ego_speed > 15.0:  # ~54 km/h
                score -= 20
            elif ego_speed > 10.0:  # ~36 km/h
                score -= 10

        # Adjust for weather
        if scene_context:
            if scene_context.weather.value in ["rainy", "foggy", "snowy"]:
                if ego_speed > 20.0:  # ~72 km/h
                    score -= 15

        # Adjust for traffic density
        if scene_analysis:
            if scene_analysis.traffic_density.value == "congested":
                if ego_speed > 10.0:
                    score -= 15
            elif scene_analysis.traffic_density.value == "heavy":
                if ego_speed > 15.0:
                    score -= 10

        return max(0.0, min(100.0, score))

    def _score_environmental_awareness(
        self,
        scene_context: Optional[SceneContext],
        scene_analysis: Optional[SceneAnalysis]
    ) -> float:
        """Score awareness and adaptation to environment (0-100)."""
        score = 100.0

        # Penalize for not being cautious in poor conditions
        if scene_context:
            if scene_context.requires_caution(scene_context):
                # Should be extra careful - this is implicitly scored by other components
                pass

        # Penalize for high danger situations
        if scene_analysis:
            if scene_analysis.danger_level.value == "danger":
                score -= 20
            elif scene_analysis.danger_level.value == "warning":
                score -= 10

            # Penalize for not being aware of vulnerable road users
            if scene_analysis.has_pedestrians or scene_analysis.has_vulnerable_road_users:
                # Extra caution needed - scored by other components
                pass

        return max(0.0, min(100.0, score))

    def _get_safety_level(self, score: float) -> SafetyLevel:
        """Convert score to safety level."""
        if score >= 90:
            return SafetyLevel.EXCELLENT
        elif score >= 75:
            return SafetyLevel.GOOD
        elif score >= 60:
            return SafetyLevel.FAIR
        elif score >= 40:
            return SafetyLevel.POOR
        else:
            return SafetyLevel.CRITICAL

    def _log_event(
        self,
        event_type: str,
        severity: str,
        score_impact: float,
        description: str
    ):
        """Log a safety event."""
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            score_impact=score_impact,
            description=description
        )

        self.safety_events.append(event)

        # Update cumulative counters
        if event_type == "near_miss":
            self.total_near_misses += 1
        elif event_type == "hard_braking":
            self.total_hard_braking += 1
        elif event_type == "lane_departure":
            self.total_lane_departures += 1
        elif event_type == "dangerous_maneuver":
            self.total_dangerous_maneuvers += 1

        logger.debug(f"Safety event: {event_type} ({severity}) - {description}")

    def _get_recent_events(self, current_time: float) -> List[SafetyEvent]:
        """Get events within the history duration."""
        # Remove old events
        cutoff_time = current_time - self.event_history_duration
        while self.safety_events and self.safety_events[0].timestamp < cutoff_time:
            self.safety_events.popleft()

        return list(self.safety_events)

    def get_average_score(self, duration_sec: Optional[float] = None) -> float:
        """
        Get average score over recent period.

        Args:
            duration_sec: Period to average over (None = all history)

        Returns:
            Average score
        """
        if not self.score_history:
            return 75.0  # Neutral

        if duration_sec is None:
            scores = [s.overall_score for s in self.score_history]
        else:
            cutoff_time = time.time() - duration_sec
            scores = [
                s.overall_score for s in self.score_history
                if s.timestamp >= cutoff_time
            ]

        return np.mean(scores) if scores else 75.0

    def get_trend(self) -> str:
        """
        Get safety score trend.

        Returns:
            "improving", "stable", or "worsening"
        """
        if len(self.score_history) < 10:
            return "stable"

        # Compare recent average to earlier average
        recent_avg = np.mean([s.overall_score for s in list(self.score_history)[-10:]])
        earlier_avg = np.mean([s.overall_score for s in list(self.score_history)[-20:-10]])

        if recent_avg > earlier_avg + 5:
            return "improving"
        elif recent_avg < earlier_avg - 5:
            return "worsening"
        else:
            return "stable"

    def get_statistics(self) -> Dict:
        """Get safety statistics."""
        current_score = self.score_history[-1] if self.score_history else None

        session_duration = time.time() - self.session_start_time

        return {
            "current_score": current_score.overall_score if current_score else None,
            "current_level": current_score.safety_level.value if current_score else None,
            "average_score": self.get_average_score(),
            "trend": self.get_trend(),
            "total_near_misses": self.total_near_misses,
            "total_hard_braking": self.total_hard_braking,
            "total_lane_departures": self.total_lane_departures,
            "total_dangerous_maneuvers": self.total_dangerous_maneuvers,
            "session_duration_min": session_duration / 60.0,
            "events_in_window": len(self._get_recent_events(time.time()))
        }

    def reset_session(self):
        """Reset session statistics."""
        self.total_near_misses = 0
        self.total_hard_braking = 0
        self.total_lane_departures = 0
        self.total_dangerous_maneuvers = 0
        self.session_start_time = time.time()
        self.score_history.clear()
        self.safety_events.clear()

        logger.info("Safety session reset")
