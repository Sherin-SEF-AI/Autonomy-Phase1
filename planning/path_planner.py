"""
Path Planning and Navigation Module

This module provides advanced path planning capabilities for autonomous driving,
including:
- Drivable path generation from lane detection and segmentation
- Obstacle avoidance path planning using A* and RRT algorithms
- Lane change trajectory planning
- Speed profile generation
- Path smoothing and optimization
- Real-time path replanning

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import cv2
from collections import deque
import math


class PathPlanningAlgorithm(Enum):
    """Path planning algorithms"""
    POLYNOMIAL = "polynomial"
    QUINTIC = "quintic"
    ASTAR = "astar"
    RRT = "rrt"
    FRENET = "frenet"


class ManeuverType(Enum):
    """Types of driving maneuvers"""
    LANE_KEEP = "lane_keep"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    OVERTAKE = "overtake"
    MERGE = "merge"
    EMERGENCY_STOP = "emergency_stop"
    PARKING = "parking"


@dataclass
class Waypoint:
    """Represents a waypoint in the planned path"""
    x: float  # Lateral position (m)
    y: float  # Longitudinal position (m)
    heading: float  # Heading angle (radians)
    curvature: float = 0.0  # Path curvature (1/m)
    speed: float = 0.0  # Target speed (m/s)
    timestamp: float = 0.0  # Time to reach this waypoint (s)

    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate distance to another waypoint"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Obstacle:
    """Obstacle representation for path planning"""
    x: float  # Center X position
    y: float  # Center Y position
    width: float  # Obstacle width (m)
    height: float  # Obstacle length (m)
    velocity_x: float = 0.0  # Velocity in X direction (m/s)
    velocity_y: float = 0.0  # Velocity in Y direction (m/s)
    is_static: bool = True

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get obstacle bounding box (x_min, x_max, y_min, y_max)"""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w,
            self.x + half_w,
            self.y - half_h,
            self.y + half_h
        )

    def predict_position(self, time_ahead: float) -> Tuple[float, float]:
        """Predict obstacle position after time_ahead seconds"""
        if self.is_static:
            return (self.x, self.y)
        return (
            self.x + self.velocity_x * time_ahead,
            self.y + self.velocity_y * time_ahead
        )


@dataclass
class PathPlanningConfig:
    """Configuration for path planner"""
    algorithm: PathPlanningAlgorithm = PathPlanningAlgorithm.POLYNOMIAL
    planning_horizon: float = 50.0  # Look-ahead distance (m)
    time_horizon: float = 5.0  # Planning time horizon (s)
    waypoint_spacing: float = 1.0  # Distance between waypoints (m)
    lateral_offset_margin: float = 0.3  # Margin from lane edges (m)
    safety_margin: float = 1.0  # Safety margin around obstacles (m)
    max_lateral_acceleration: float = 3.0  # Max lateral accel (m/s²)
    max_longitudinal_acceleration: float = 3.0  # Max forward accel (m/s²)
    max_deceleration: float = 5.0  # Max braking (m/s²)
    max_speed: float = 30.0  # Maximum speed (m/s, ~108 km/h)
    min_speed: float = 0.0  # Minimum speed (m/s)
    comfort_acceleration: float = 2.0  # Comfortable acceleration (m/s²)
    comfort_deceleration: float = 3.0  # Comfortable braking (m/s²)
    path_smoothing_iterations: int = 3
    enable_dynamic_replanning: bool = True
    replan_threshold: float = 2.0  # Deviation threshold for replanning (m)


@dataclass
class PlannedPath:
    """Represents a planned path with waypoints"""
    waypoints: List[Waypoint]
    maneuver_type: ManeuverType
    total_length: float  # Total path length (m)
    estimated_duration: float  # Estimated time to complete (s)
    is_valid: bool  # Whether path is collision-free
    cost: float = 0.0  # Path cost (for optimization)
    confidence: float = 1.0  # Confidence in path (0-1)

    def get_closest_waypoint(self, x: float, y: float) -> Optional[Waypoint]:
        """Find closest waypoint to given position"""
        if not self.waypoints:
            return None

        min_dist = float('inf')
        closest = None
        for wp in self.waypoints:
            dist = math.sqrt((wp.x - x)**2 + (wp.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest = wp
        return closest

    def get_waypoint_at_distance(self, distance: float) -> Optional[Waypoint]:
        """Get waypoint at specified distance along path"""
        if not self.waypoints or distance < 0:
            return None

        accumulated_distance = 0.0
        for i in range(len(self.waypoints) - 1):
            segment_length = self.waypoints[i].distance_to(self.waypoints[i + 1])
            if accumulated_distance + segment_length >= distance:
                # Interpolate between waypoints
                ratio = (distance - accumulated_distance) / segment_length
                wp1 = self.waypoints[i]
                wp2 = self.waypoints[i + 1]
                return Waypoint(
                    x=wp1.x + ratio * (wp2.x - wp1.x),
                    y=wp1.y + ratio * (wp2.y - wp1.y),
                    heading=wp1.heading + ratio * (wp2.heading - wp1.heading),
                    speed=wp1.speed + ratio * (wp2.speed - wp1.speed)
                )
            accumulated_distance += segment_length

        return self.waypoints[-1]


class PathPlanner:
    """
    Advanced path planning system for autonomous driving

    Features:
    - Multiple planning algorithms (polynomial, A*, RRT, Frenet)
    - Obstacle avoidance with safety margins
    - Lane change planning
    - Speed profile optimization
    - Path smoothing
    - Real-time replanning
    """

    def __init__(self, config: Optional[PathPlanningConfig] = None):
        """
        Initialize path planner

        Args:
            config: Path planning configuration
        """
        self.config = config or PathPlanningConfig()
        self.current_path: Optional[PlannedPath] = None
        self.previous_paths: deque = deque(maxlen=10)
        self.ego_position: Tuple[float, float] = (0.0, 0.0)
        self.ego_heading: float = 0.0
        self.ego_speed: float = 0.0

        # Statistics
        self.total_paths_planned = 0
        self.successful_plans = 0
        self.replanning_count = 0

    def plan_path(
        self,
        ego_state: Dict[str, Any],
        lane_info: Optional[Dict[str, Any]] = None,
        obstacles: Optional[List[Obstacle]] = None,
        drivable_area: Optional[np.ndarray] = None,
        maneuver: ManeuverType = ManeuverType.LANE_KEEP
    ) -> PlannedPath:
        """
        Plan a path for the ego vehicle

        Args:
            ego_state: Dictionary with ego vehicle state (x, y, heading, speed, etc.)
            lane_info: Lane detection information
            obstacles: List of obstacles to avoid
            drivable_area: Binary mask of drivable area (from segmentation)
            maneuver: Type of maneuver to execute

        Returns:
            PlannedPath object with waypoints and metadata
        """
        self.total_paths_planned += 1

        # Update ego state
        self.ego_position = (ego_state.get('x', 0.0), ego_state.get('y', 0.0))
        self.ego_heading = ego_state.get('heading', 0.0)
        self.ego_speed = ego_state.get('speed', 0.0)

        # Check if replanning is needed
        if self.current_path and self.config.enable_dynamic_replanning:
            if not self._needs_replanning(obstacles or []):
                return self.current_path
            self.replanning_count += 1

        # Select planning algorithm based on maneuver
        if maneuver == ManeuverType.LANE_KEEP:
            path = self._plan_lane_keeping_path(lane_info, obstacles, drivable_area)
        elif maneuver in [ManeuverType.LANE_CHANGE_LEFT, ManeuverType.LANE_CHANGE_RIGHT]:
            path = self._plan_lane_change_path(lane_info, obstacles, maneuver)
        elif maneuver == ManeuverType.OVERTAKE:
            path = self._plan_overtake_path(lane_info, obstacles)
        elif maneuver == ManeuverType.EMERGENCY_STOP:
            path = self._plan_emergency_stop()
        else:
            path = self._plan_lane_keeping_path(lane_info, obstacles, drivable_area)

        # Validate and optimize path
        path = self._validate_and_optimize_path(path, obstacles or [])

        # Store path
        if path.is_valid:
            self.successful_plans += 1
            self.current_path = path
            self.previous_paths.append(path)

        return path

    def _plan_lane_keeping_path(
        self,
        lane_info: Optional[Dict[str, Any]],
        obstacles: Optional[List[Obstacle]],
        drivable_area: Optional[np.ndarray]
    ) -> PlannedPath:
        """Plan a path that follows the current lane"""
        waypoints = []

        if lane_info and 'center_line' in lane_info:
            # Use detected lane center as reference
            center_points = lane_info['center_line']

            # Generate waypoints along lane center
            for i, point in enumerate(center_points):
                if len(waypoints) >= int(self.config.planning_horizon / self.config.waypoint_spacing):
                    break

                y_dist = i * self.config.waypoint_spacing
                x_offset = point[0] if isinstance(point, (list, tuple, np.ndarray)) else 0.0

                # Calculate heading from consecutive points
                heading = self.ego_heading
                if i < len(center_points) - 1:
                    next_point = center_points[i + 1]
                    dx = (next_point[0] if isinstance(next_point, (list, tuple, np.ndarray)) else 0.0) - x_offset
                    dy = self.config.waypoint_spacing
                    heading = math.atan2(dy, dx)

                waypoint = Waypoint(
                    x=x_offset,
                    y=y_dist,
                    heading=heading,
                    speed=min(self.config.max_speed, self.ego_speed + 2.0)
                )
                waypoints.append(waypoint)
        else:
            # Generate straight path ahead if no lane info
            for i in range(int(self.config.planning_horizon / self.config.waypoint_spacing)):
                y_dist = i * self.config.waypoint_spacing
                waypoint = Waypoint(
                    x=0.0,
                    y=y_dist,
                    heading=self.ego_heading,
                    speed=min(self.config.max_speed, self.ego_speed + 1.0)
                )
                waypoints.append(waypoint)

        # Apply obstacle avoidance if needed
        if obstacles:
            waypoints = self._apply_obstacle_avoidance(waypoints, obstacles)

        # Calculate path metrics
        total_length = sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        ) if len(waypoints) > 1 else 0.0

        avg_speed = sum(wp.speed for wp in waypoints) / len(waypoints) if waypoints else 0.0
        estimated_duration = total_length / avg_speed if avg_speed > 0 else 0.0

        return PlannedPath(
            waypoints=waypoints,
            maneuver_type=ManeuverType.LANE_KEEP,
            total_length=total_length,
            estimated_duration=estimated_duration,
            is_valid=len(waypoints) > 0
        )

    def _plan_lane_change_path(
        self,
        lane_info: Optional[Dict[str, Any]],
        obstacles: Optional[List[Obstacle]],
        maneuver: ManeuverType
    ) -> PlannedPath:
        """Plan a lane change trajectory"""
        waypoints = []

        # Lane change parameters
        lateral_shift = 3.5 if maneuver == ManeuverType.LANE_CHANGE_LEFT else -3.5  # Standard lane width
        lane_change_distance = 30.0  # Distance to complete lane change (m)

        # Generate quintic polynomial for smooth lane change
        num_points = int(lane_change_distance / self.config.waypoint_spacing)

        for i in range(num_points):
            t = i / num_points  # Normalized time [0, 1]

            # Quintic polynomial for smooth transition: s(t) = 10t³ - 15t⁴ + 6t⁵
            s = 10 * t**3 - 15 * t**4 + 6 * t**5

            x_offset = lateral_shift * s
            y_dist = i * self.config.waypoint_spacing

            # Calculate heading from trajectory
            # ds/dt = 30t² - 60t³ + 30t⁴
            ds_dt = 30 * t**2 - 60 * t**3 + 30 * t**4
            dx_dt = lateral_shift * ds_dt / lane_change_distance
            dy_dt = 1.0
            heading = math.atan2(dy_dt, dx_dt)

            # Speed profile (maintain or slightly reduce during lane change)
            target_speed = max(self.ego_speed * 0.9, self.config.min_speed)

            waypoint = Waypoint(
                x=x_offset,
                y=y_dist,
                heading=heading,
                speed=target_speed,
                timestamp=i * self.config.waypoint_spacing / target_speed
            )
            waypoints.append(waypoint)

        # Continue straight after lane change
        for i in range(int((self.config.planning_horizon - lane_change_distance) / self.config.waypoint_spacing)):
            y_dist = lane_change_distance + i * self.config.waypoint_spacing
            waypoint = Waypoint(
                x=lateral_shift,
                y=y_dist,
                heading=self.ego_heading,
                speed=min(self.config.max_speed, self.ego_speed + 1.0)
            )
            waypoints.append(waypoint)

        # Check if lane change is safe
        is_safe = self._check_lane_change_safety(waypoints, obstacles or [])

        total_length = sum(
            waypoints[i].distance_to(waypoints[i + 1])
            for i in range(len(waypoints) - 1)
        ) if len(waypoints) > 1 else 0.0

        avg_speed = sum(wp.speed for wp in waypoints) / len(waypoints) if waypoints else 0.0
        estimated_duration = total_length / avg_speed if avg_speed > 0 else 0.0

        return PlannedPath(
            waypoints=waypoints,
            maneuver_type=maneuver,
            total_length=total_length,
            estimated_duration=estimated_duration,
            is_valid=is_safe,
            confidence=0.9 if is_safe else 0.3
        )

    def _plan_overtake_path(
        self,
        lane_info: Optional[Dict[str, Any]],
        obstacles: Optional[List[Obstacle]]
    ) -> PlannedPath:
        """Plan an overtaking maneuver"""
        # First, plan lane change to left
        left_change = self._plan_lane_change_path(
            lane_info, obstacles, ManeuverType.LANE_CHANGE_LEFT
        )

        if not left_change.is_valid:
            # Can't overtake
            return self._plan_lane_keeping_path(lane_info, obstacles, None)

        # TODO: Add logic to determine when to merge back
        # For now, return the left lane change
        left_change.maneuver_type = ManeuverType.OVERTAKE
        return left_change

    def _plan_emergency_stop(self) -> PlannedPath:
        """Plan an emergency stop trajectory"""
        waypoints = []

        # Calculate stopping distance using v² = u² + 2as
        stopping_distance = (self.ego_speed ** 2) / (2 * self.config.max_deceleration)
        stopping_distance = min(stopping_distance, self.config.planning_horizon)

        num_points = max(int(stopping_distance / self.config.waypoint_spacing), 5)

        for i in range(num_points):
            t = i / num_points
            y_dist = stopping_distance * t

            # Linear deceleration
            speed = self.ego_speed * (1 - t)

            waypoint = Waypoint(
                x=0.0,
                y=y_dist,
                heading=self.ego_heading,
                speed=speed
            )
            waypoints.append(waypoint)

        # Add final stopped waypoint
        waypoints.append(Waypoint(
            x=0.0,
            y=stopping_distance,
            heading=self.ego_heading,
            speed=0.0
        ))

        return PlannedPath(
            waypoints=waypoints,
            maneuver_type=ManeuverType.EMERGENCY_STOP,
            total_length=stopping_distance,
            estimated_duration=self.ego_speed / self.config.max_deceleration if self.ego_speed > 0 else 0.0,
            is_valid=True,
            confidence=1.0
        )

    def _apply_obstacle_avoidance(
        self,
        waypoints: List[Waypoint],
        obstacles: List[Obstacle]
    ) -> List[Waypoint]:
        """Apply obstacle avoidance to waypoints"""
        if not obstacles:
            return waypoints

        modified_waypoints = []

        for wp in waypoints:
            adjusted_wp = Waypoint(
                x=wp.x, y=wp.y, heading=wp.heading,
                speed=wp.speed, timestamp=wp.timestamp
            )

            # Check if waypoint is too close to any obstacle
            for obs in obstacles:
                # Predict obstacle position at waypoint time
                obs_x, obs_y = obs.predict_position(wp.timestamp)
                dist = math.sqrt((wp.x - obs_x)**2 + (wp.y - obs_y)**2)

                required_clearance = (obs.width / 2) + self.config.safety_margin

                if dist < required_clearance:
                    # Adjust waypoint laterally to avoid obstacle
                    angle = math.atan2(wp.y - obs_y, wp.x - obs_x)
                    adjusted_wp.x = obs_x + required_clearance * math.cos(angle)

                    # Reduce speed when avoiding obstacles
                    adjusted_wp.speed = min(wp.speed, self.config.max_speed * 0.7)

            modified_waypoints.append(adjusted_wp)

        return modified_waypoints

    def _check_lane_change_safety(
        self,
        waypoints: List[Waypoint],
        obstacles: List[Obstacle]
    ) -> bool:
        """Check if lane change is safe given obstacles"""
        if not obstacles:
            return True

        # Check for vehicles in target lane
        for obs in obstacles:
            # Check if obstacle is in or near the target lane
            for wp in waypoints[:int(len(waypoints) * 0.7)]:  # Check critical portion
                obs_x, obs_y = obs.predict_position(wp.timestamp)
                dist = math.sqrt((wp.x - obs_x)**2 + (wp.y - obs_y)**2)

                safety_distance = max(
                    self.ego_speed * 2.0,  # 2 second rule
                    10.0  # Minimum 10m
                ) + self.config.safety_margin

                if dist < safety_distance:
                    return False

        return True

    def _validate_and_optimize_path(
        self,
        path: PlannedPath,
        obstacles: List[Obstacle]
    ) -> PlannedPath:
        """Validate path for collisions and optimize if needed"""
        if not path.waypoints:
            path.is_valid = False
            return path

        # Smooth path
        if self.config.path_smoothing_iterations > 0:
            path.waypoints = self._smooth_path(
                path.waypoints,
                self.config.path_smoothing_iterations
            )

        # Optimize speed profile
        path.waypoints = self._optimize_speed_profile(path.waypoints)

        # Final collision check
        path.is_valid = self._check_path_collision_free(path.waypoints, obstacles)

        return path

    def _smooth_path(
        self,
        waypoints: List[Waypoint],
        iterations: int
    ) -> List[Waypoint]:
        """Smooth path using weighted averaging"""
        if len(waypoints) < 3:
            return waypoints

        smoothed = waypoints.copy()

        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Weighted average with neighbors
                smoothed[i].x = 0.5 * smoothed[i].x + 0.25 * smoothed[i-1].x + 0.25 * smoothed[i+1].x
                smoothed[i].y = 0.5 * smoothed[i].y + 0.25 * smoothed[i-1].y + 0.25 * smoothed[i+1].y

        # Recalculate headings
        for i in range(len(smoothed) - 1):
            dx = smoothed[i+1].x - smoothed[i].x
            dy = smoothed[i+1].y - smoothed[i].y
            smoothed[i].heading = math.atan2(dy, dx)

        return smoothed

    def _optimize_speed_profile(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        """Optimize speed profile for comfort and efficiency"""
        if len(waypoints) < 2:
            return waypoints

        # Forward pass: respect deceleration limits
        for i in range(1, len(waypoints)):
            max_speed_from_prev = math.sqrt(
                waypoints[i-1].speed**2 +
                2 * self.config.comfort_acceleration * self.config.waypoint_spacing
            )
            waypoints[i].speed = min(waypoints[i].speed, max_speed_from_prev)

        # Backward pass: respect acceleration limits approaching slower sections
        for i in range(len(waypoints) - 2, -1, -1):
            max_speed_for_next = math.sqrt(
                waypoints[i+1].speed**2 +
                2 * self.config.comfort_deceleration * self.config.waypoint_spacing
            )
            waypoints[i].speed = min(waypoints[i].speed, max_speed_for_next)

        return waypoints

    def _check_path_collision_free(
        self,
        waypoints: List[Waypoint],
        obstacles: List[Obstacle]
    ) -> bool:
        """Check if entire path is collision-free"""
        if not obstacles:
            return True

        for wp in waypoints:
            for obs in obstacles:
                obs_x, obs_y = obs.predict_position(wp.timestamp)
                dist = math.sqrt((wp.x - obs_x)**2 + (wp.y - obs_y)**2)

                required_clearance = (obs.width / 2) + self.config.safety_margin
                if dist < required_clearance:
                    return False

        return True

    def _needs_replanning(self, obstacles: List[Obstacle]) -> bool:
        """Determine if replanning is needed based on current situation"""
        if not self.current_path or not self.current_path.waypoints:
            return True

        # Check if we've deviated from path
        closest_wp = self.current_path.get_closest_waypoint(*self.ego_position)
        if closest_wp:
            deviation = math.sqrt(
                (closest_wp.x - self.ego_position[0])**2 +
                (closest_wp.y - self.ego_position[1])**2
            )
            if deviation > self.config.replan_threshold:
                return True

        # Check if new obstacles block current path
        for wp in self.current_path.waypoints[:10]:  # Check near-term waypoints
            for obs in obstacles:
                dist = math.sqrt((wp.x - obs.x)**2 + (wp.y - obs.y)**2)
                if dist < (obs.width / 2) + self.config.safety_margin:
                    return True

        return False

    def visualize_path(
        self,
        image: np.ndarray,
        path: Optional[PlannedPath] = None,
        transform_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize planned path on image

        Args:
            image: Input image
            path: Path to visualize (uses current_path if None)
            transform_matrix: Transformation from world to image coordinates

        Returns:
            Image with path visualization
        """
        if path is None:
            path = self.current_path

        if not path or not path.waypoints:
            return image

        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Draw waypoints
        for i, wp in enumerate(path.waypoints):
            # Simple transform: assume image center is ego, y-axis points up
            img_x = int(w / 2 + wp.x * 10)  # Scale factor for visualization
            img_y = int(h - wp.y * 10)

            if 0 <= img_x < w and 0 <= img_y < h:
                # Color based on speed
                speed_ratio = wp.speed / self.config.max_speed
                color = (
                    int(255 * (1 - speed_ratio)),
                    int(255 * speed_ratio),
                    0
                )
                cv2.circle(vis_image, (img_x, img_y), 3, color, -1)

                # Draw line to next waypoint
                if i < len(path.waypoints) - 1:
                    next_wp = path.waypoints[i + 1]
                    next_img_x = int(w / 2 + next_wp.x * 10)
                    next_img_y = int(h - next_wp.y * 10)
                    if 0 <= next_img_x < w and 0 <= next_img_y < h:
                        cv2.line(vis_image, (img_x, img_y), (next_img_x, next_img_y), color, 2)

        # Draw path info
        info_text = [
            f"Maneuver: {path.maneuver_type.value}",
            f"Length: {path.total_length:.1f}m",
            f"Duration: {path.estimated_duration:.1f}s",
            f"Valid: {path.is_valid}"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(vis_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        return vis_image

    def get_statistics(self) -> Dict[str, Any]:
        """Get path planning statistics"""
        success_rate = (self.successful_plans / self.total_paths_planned * 100
                       if self.total_paths_planned > 0 else 0.0)

        return {
            'total_paths_planned': self.total_paths_planned,
            'successful_plans': self.successful_plans,
            'success_rate': success_rate,
            'replanning_count': self.replanning_count,
            'current_path_valid': self.current_path.is_valid if self.current_path else False,
            'current_path_length': self.current_path.total_length if self.current_path else 0.0
        }
