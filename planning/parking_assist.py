"""
Parking Assist System

This module provides advanced parking assistance with trajectory planning,
including:
- Parking space detection
- Multiple parking modes (parallel, perpendicular, angled)
- Collision-free trajectory planning
- Real-time trajectory visualization
- Steering angle calculation
- Multi-point turn planning
- Parking spot quality assessment
- Automated parking guidance

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math
from collections import deque


class ParkingMode(Enum):
    """Parking maneuver types"""
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    ANGLED_45 = "angled_45"
    ANGLED_60 = "angled_60"


class ParkingPhase(Enum):
    """Parking maneuver phases"""
    SEARCHING = "searching"
    APPROACHING = "approaching"
    ALIGNING = "aligning"
    ENTERING = "entering"
    ADJUSTING = "adjusting"
    PARKED = "parked"
    EXITING = "exiting"


class SpotQuality(Enum):
    """Parking spot quality ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    DIFFICULT = "difficult"
    UNSUITABLE = "unsuitable"


@dataclass
class ParkingSpot:
    """Represents a detected parking space"""
    corners: List[Tuple[float, float]]  # 4 corners (x, y) in meters
    mode: ParkingMode
    quality: SpotQuality
    length: float  # meters
    width: float  # meters
    confidence: float
    is_occupied: bool = False
    obstacles_nearby: int = 0


@dataclass
class VehicleDimensions:
    """Vehicle physical dimensions"""
    length: float = 4.5  # meters
    width: float = 1.8  # meters
    wheelbase: float = 2.7  # meters
    front_overhang: float = 0.9  # meters
    rear_overhang: float = 0.9  # meters
    turning_radius: float = 5.5  # meters
    min_turning_radius: float = 5.0  # meters
    max_steering_angle: float = 35.0  # degrees


@dataclass
class ParkingWaypoint:
    """Waypoint in parking trajectory"""
    x: float  # meters
    y: float  # meters
    heading: float  # radians
    steering_angle: float  # degrees
    speed: float  # m/s
    gear: str = "D"  # D, R, N, P


@dataclass
class ParkingTrajectory:
    """Complete parking trajectory"""
    waypoints: List[ParkingWaypoint]
    mode: ParkingMode
    total_distance: float
    estimated_time: float  # seconds
    num_maneuvers: int  # number of direction changes
    is_feasible: bool
    clearance_min: float  # minimum clearance to obstacles (meters)


@dataclass
class ParkingAssistConfig:
    """Configuration for parking assist system"""
    vehicle: VehicleDimensions = field(default_factory=VehicleDimensions)

    # Space requirements (margins beyond vehicle dimensions)
    parallel_length_margin: float = 1.5  # meters
    parallel_width_margin: float = 0.6  # meters
    perpendicular_length_margin: float = 0.8  # meters
    perpendicular_width_margin: float = 0.6  # meters

    # Safety margins
    obstacle_clearance: float = 0.3  # meters
    wall_clearance: float = 0.2  # meters

    # Trajectory planning
    waypoint_spacing: float = 0.2  # meters
    max_maneuvers: int = 5  # max direction changes
    parking_speed: float = 0.5  # m/s (very slow)

    # Detection
    spot_confidence_threshold: float = 0.7
    max_spot_search_distance: float = 50.0  # meters


class ParkingAssistSystem:
    """
    Parking Assist System

    Features:
    - Automatic parking space detection
    - Multi-mode parking (parallel, perpendicular, angled)
    - Collision-free trajectory planning
    - Multi-point turn trajectories
    - Real-time visualization
    - Steering guidance
    - Distance and clearance monitoring
    """

    def __init__(self, config: Optional[ParkingAssistConfig] = None):
        """
        Initialize parking assist system

        Args:
            config: Parking assist configuration
        """
        self.config = config or ParkingAssistConfig()
        self.detected_spots: List[ParkingSpot] = []
        self.target_spot: Optional[ParkingSpot] = None
        self.current_trajectory: Optional[ParkingTrajectory] = None
        self.current_phase = ParkingPhase.SEARCHING

        # Statistics
        self.total_spots_detected = 0
        self.successful_parking_plans = 0

    def detect_parking_spots(
        self,
        obstacles: List[Dict[str, Any]],
        drivable_area: Optional[np.ndarray] = None,
        lane_markings: Optional[List[Any]] = None
    ) -> List[ParkingSpot]:
        """
        Detect available parking spaces

        Args:
            obstacles: List of detected obstacles with positions
            drivable_area: Drivable area segmentation mask
            lane_markings: Detected lane markings

        Returns:
            List of detected parking spots
        """
        spots = []

        # Method 1: Detect gaps between parked vehicles
        if obstacles:
            spots.extend(self._detect_gaps_between_vehicles(obstacles))

        # Method 2: Detect marked parking spaces from lane markings
        if lane_markings:
            spots.extend(self._detect_marked_spaces(lane_markings))

        # Method 3: Detect spaces along walls/curbs
        if drivable_area is not None:
            spots.extend(self._detect_wall_spaces(drivable_area))

        # Filter and assess quality
        valid_spots = []
        for spot in spots:
            if spot.confidence >= self.config.spot_confidence_threshold:
                spot.quality = self._assess_spot_quality(spot, obstacles)
                valid_spots.append(spot)
                self.total_spots_detected += 1

        self.detected_spots = valid_spots
        return valid_spots

    def _detect_gaps_between_vehicles(
        self,
        obstacles: List[Dict[str, Any]]
    ) -> List[ParkingSpot]:
        """Detect parking spaces as gaps between vehicles"""
        spots = []

        # Filter for parked vehicles (stationary cars along the side)
        parked_vehicles = [
            obs for obs in obstacles
            if obs.get('class') in ['car', 'truck', 'bus'] and
            obs.get('velocity', 0) < 0.1  # Nearly stationary
        ]

        if len(parked_vehicles) < 2:
            return spots

        # Sort by position (e.g., along road)
        parked_vehicles.sort(key=lambda v: v.get('position', (0, 0))[1])

        # Check gaps between consecutive vehicles
        for i in range(len(parked_vehicles) - 1):
            v1 = parked_vehicles[i]
            v2 = parked_vehicles[i + 1]

            pos1 = v1.get('position', (0, 0))
            pos2 = v2.get('position', (0, 0))

            # Calculate gap size
            gap_length = abs(pos2[1] - pos1[1]) - v1.get('length', 4.5) / 2 - v2.get('length', 4.5) / 2

            # Determine parking mode (parallel if vehicles are side-by-side)
            lateral_offset = abs(pos2[0] - pos1[0])

            if lateral_offset < 2.0:  # Vehicles in same lane
                mode = ParkingMode.PARALLEL
                required_length = self.config.vehicle.length + self.config.parallel_length_margin
            else:
                mode = ParkingMode.PERPENDICULAR
                required_length = self.config.vehicle.length + self.config.perpendicular_length_margin

            # Check if gap is large enough
            if gap_length >= required_length:
                # Create spot
                spot_center_x = (pos1[0] + pos2[0]) / 2
                spot_center_y = (pos1[1] + pos2[1]) / 2

                # Define corners (simplified rectangular spot)
                spot_width = self.config.vehicle.width + self.config.parallel_width_margin
                spot_length = gap_length

                corners = [
                    (spot_center_x - spot_width/2, spot_center_y - spot_length/2),
                    (spot_center_x + spot_width/2, spot_center_y - spot_length/2),
                    (spot_center_x + spot_width/2, spot_center_y + spot_length/2),
                    (spot_center_x - spot_width/2, spot_center_y + spot_length/2),
                ]

                spot = ParkingSpot(
                    corners=corners,
                    mode=mode,
                    quality=SpotQuality.GOOD,
                    length=spot_length,
                    width=spot_width,
                    confidence=0.8,
                    obstacles_nearby=2
                )
                spots.append(spot)

        return spots

    def _detect_marked_spaces(
        self,
        lane_markings: List[Any]
    ) -> List[ParkingSpot]:
        """Detect parking spaces from painted markings"""
        # Simplified: would analyze lane markings to find perpendicular
        # parking space boundaries
        # TODO: Implement proper marking analysis
        return []

    def _detect_wall_spaces(
        self,
        drivable_area: np.ndarray
    ) -> List[ParkingSpot]:
        """Detect parking spaces along walls or edges"""
        # Simplified: would analyze edges of drivable area
        # TODO: Implement edge-based detection
        return []

    def _assess_spot_quality(
        self,
        spot: ParkingSpot,
        obstacles: List[Dict[str, Any]]
    ) -> SpotQuality:
        """Assess quality/difficulty of parking spot"""
        score = 100.0

        # Size adequacy
        size_ratio = spot.length / (self.config.vehicle.length + self.config.parallel_length_margin)
        if size_ratio < 1.1:
            score -= 30
        elif size_ratio < 1.2:
            score -= 15

        # Nearby obstacles
        score -= spot.obstacles_nearby * 10

        # Mode difficulty
        if spot.mode == ParkingMode.PARALLEL:
            score -= 10  # Parallel is harder
        elif spot.mode in [ParkingMode.ANGLED_45, ParkingMode.ANGLED_60]:
            score += 10  # Angled is easier

        # Classify quality
        if score >= 80:
            return SpotQuality.EXCELLENT
        elif score >= 65:
            return SpotQuality.GOOD
        elif score >= 50:
            return SpotQuality.ACCEPTABLE
        elif score >= 30:
            return SpotQuality.DIFFICULT
        else:
            return SpotQuality.UNSUITABLE

    def plan_parking_trajectory(
        self,
        target_spot: ParkingSpot,
        current_position: Tuple[float, float, float],  # x, y, heading
        obstacles: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[ParkingTrajectory]:
        """
        Plan parking trajectory to target spot

        Args:
            target_spot: Target parking space
            current_position: Current vehicle position (x, y, heading_rad)
            obstacles: List of obstacles to avoid

        Returns:
            ParkingTrajectory or None if not feasible
        """
        self.target_spot = target_spot

        # Select planning method based on parking mode
        if target_spot.mode == ParkingMode.PARALLEL:
            trajectory = self._plan_parallel_parking(target_spot, current_position, obstacles)
        elif target_spot.mode == ParkingMode.PERPENDICULAR:
            trajectory = self._plan_perpendicular_parking(target_spot, current_position, obstacles)
        elif target_spot.mode in [ParkingMode.ANGLED_45, ParkingMode.ANGLED_60]:
            trajectory = self._plan_angled_parking(target_spot, current_position, obstacles)
        else:
            return None

        if trajectory and trajectory.is_feasible:
            self.current_trajectory = trajectory
            self.successful_parking_plans += 1

        return trajectory

    def _plan_parallel_parking(
        self,
        spot: ParkingSpot,
        current_pos: Tuple[float, float, float],
        obstacles: Optional[List[Dict[str, Any]]]
    ) -> ParkingTrajectory:
        """
        Plan parallel parking maneuver

        Classic 3-point parallel parking:
        1. Pull alongside front vehicle
        2. Reverse while turning into spot
        3. Straighten and adjust position
        """
        waypoints = []
        curr_x, curr_y, curr_heading = current_pos

        # Target position: center of parking spot
        spot_center_x = sum(c[0] for c in spot.corners) / 4
        spot_center_y = sum(c[1] for c in spot.corners) / 4

        # Phase 1: Approach - pull alongside front vehicle
        approach_x = spot_center_x - 1.0  # Lateral offset
        approach_y = spot_center_y + spot.length / 2 + 2.0

        num_approach_points = int(np.linalg.norm([approach_x - curr_x, approach_y - curr_y]) /
                                 self.config.waypoint_spacing)

        for i in range(num_approach_points):
            t = i / num_approach_points
            x = curr_x + t * (approach_x - curr_x)
            y = curr_y + t * (approach_y - curr_y)
            heading = math.atan2(approach_y - curr_y, approach_x - curr_x)

            waypoints.append(ParkingWaypoint(
                x=x, y=y, heading=heading,
                steering_angle=0.0,
                speed=self.config.parking_speed,
                gear="D"
            ))

        # Phase 2: Reverse into spot with right turn
        reverse_start = (approach_x, approach_y, 0.0)
        reverse_end = (spot_center_x + 0.3, spot_center_y, -math.pi / 6)  # Angled

        num_reverse_points = int(spot.length / self.config.waypoint_spacing)

        for i in range(num_reverse_points):
            t = i / num_reverse_points
            x = reverse_start[0] + t * (reverse_end[0] - reverse_start[0])
            y = reverse_start[1] + t * (reverse_end[1] - reverse_start[1])
            heading = reverse_start[2] + t * (reverse_end[2] - reverse_start[2])

            # Calculate steering angle for arc
            steering = -self.config.vehicle.max_steering_angle * (1 - t)

            waypoints.append(ParkingWaypoint(
                x=x, y=y, heading=heading,
                steering_angle=steering,
                speed=self.config.parking_speed,
                gear="R"
            ))

        # Phase 3: Straighten
        final_x = spot_center_x
        final_y = spot_center_y
        final_heading = 0.0

        num_adjust_points = 5
        for i in range(num_adjust_points):
            t = i / num_adjust_points
            x = reverse_end[0] + t * (final_x - reverse_end[0])
            y = reverse_end[1] + t * (final_y - reverse_end[1])
            heading = reverse_end[2] + t * (final_heading - reverse_end[2])

            waypoints.append(ParkingWaypoint(
                x=x, y=y, heading=heading,
                steering_angle=0.0,
                speed=self.config.parking_speed * 0.5,
                gear="R"
            ))

        # Final parked position
        waypoints.append(ParkingWaypoint(
            x=final_x, y=final_y, heading=final_heading,
            steering_angle=0.0, speed=0.0, gear="P"
        ))

        # Check feasibility
        is_feasible = self._check_trajectory_feasible(waypoints, obstacles)

        total_distance = sum(
            math.sqrt((waypoints[i+1].x - waypoints[i].x)**2 +
                     (waypoints[i+1].y - waypoints[i].y)**2)
            for i in range(len(waypoints) - 1)
        )

        return ParkingTrajectory(
            waypoints=waypoints,
            mode=ParkingMode.PARALLEL,
            total_distance=total_distance,
            estimated_time=total_distance / self.config.parking_speed,
            num_maneuvers=2,  # Forward then reverse
            is_feasible=is_feasible,
            clearance_min=self.config.obstacle_clearance
        )

    def _plan_perpendicular_parking(
        self,
        spot: ParkingSpot,
        current_pos: Tuple[float, float, float],
        obstacles: Optional[List[Dict[str, Any]]]
    ) -> ParkingTrajectory:
        """Plan perpendicular parking maneuver"""
        waypoints = []
        curr_x, curr_y, curr_heading = current_pos

        # Target: center of spot
        spot_center_x = sum(c[0] for c in spot.corners) / 4
        spot_center_y = sum(c[1] for c in spot.corners) / 4

        # Approach: drive to position in front of spot
        approach_y = spot_center_y - (spot.length / 2 + 3.0)
        approach_x = spot_center_x + 2.0  # Offset for turning room

        # Generate approach waypoints
        num_approach = int(math.sqrt((approach_x - curr_x)**2 + (approach_y - curr_y)**2) /
                          self.config.waypoint_spacing)

        for i in range(num_approach):
            t = i / num_approach
            x = curr_x + t * (approach_x - curr_x)
            y = curr_y + t * (approach_y - curr_y)
            heading = math.atan2(approach_y - curr_y, approach_x - curr_x)

            waypoints.append(ParkingWaypoint(
                x=x, y=y, heading=heading,
                steering_angle=0.0,
                speed=self.config.parking_speed,
                gear="D"
            ))

        # Turn into spot
        turn_radius = self.config.vehicle.turning_radius
        turn_center_x = approach_x - turn_radius
        turn_center_y = approach_y

        num_turn = 20
        for i in range(num_turn):
            angle = math.pi / 2 * (i / num_turn)
            x = turn_center_x + turn_radius * math.cos(angle)
            y = turn_center_y + turn_radius * math.sin(angle)
            heading = angle + math.pi / 2

            steering = self.config.vehicle.max_steering_angle * (1 - i / num_turn)

            waypoints.append(ParkingWaypoint(
                x=x, y=y, heading=heading,
                steering_angle=steering,
                speed=self.config.parking_speed * 0.7,
                gear="D"
            ))

        # Drive into spot
        num_enter = int(spot.length / self.config.waypoint_spacing)
        for i in range(num_enter):
            y = spot_center_y - spot.length / 2 + (i / num_enter) * spot.length
            waypoints.append(ParkingWaypoint(
                x=spot_center_x, y=y, heading=math.pi / 2,
                steering_angle=0.0,
                speed=self.config.parking_speed * 0.5,
                gear="D"
            ))

        # Final position
        waypoints.append(ParkingWaypoint(
            x=spot_center_x, y=spot_center_y, heading=math.pi / 2,
            steering_angle=0.0, speed=0.0, gear="P"
        ))

        is_feasible = self._check_trajectory_feasible(waypoints, obstacles)

        total_distance = sum(
            math.sqrt((waypoints[i+1].x - waypoints[i].x)**2 +
                     (waypoints[i+1].y - waypoints[i].y)**2)
            for i in range(len(waypoints) - 1)
        )

        return ParkingTrajectory(
            waypoints=waypoints,
            mode=ParkingMode.PERPENDICULAR,
            total_distance=total_distance,
            estimated_time=total_distance / self.config.parking_speed,
            num_maneuvers=1,
            is_feasible=is_feasible,
            clearance_min=self.config.obstacle_clearance
        )

    def _plan_angled_parking(
        self,
        spot: ParkingSpot,
        current_pos: Tuple[float, float, float],
        obstacles: Optional[List[Dict[str, Any]]]
    ) -> ParkingTrajectory:
        """Plan angled parking maneuver (easiest)"""
        # Similar to perpendicular but with angled entry
        # For simplicity, reuse perpendicular with adjusted angles
        return self._plan_perpendicular_parking(spot, current_pos, obstacles)

    def _check_trajectory_feasible(
        self,
        waypoints: List[ParkingWaypoint],
        obstacles: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Check if trajectory is collision-free"""
        if not obstacles:
            return True

        # Check each waypoint for collisions
        for wp in waypoints:
            vehicle_bbox = self._get_vehicle_bbox_at_waypoint(wp)

            for obs in obstacles:
                obs_pos = obs.get('position', (0, 0))
                obs_size = obs.get('size', (2.0, 2.0))

                # Simple rectangle collision check
                if self._rectangles_collide(vehicle_bbox, (obs_pos, obs_size)):
                    return False

        return True

    def _get_vehicle_bbox_at_waypoint(
        self,
        waypoint: ParkingWaypoint
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get vehicle bounding box at waypoint"""
        # Simplified: axis-aligned box
        half_length = self.config.vehicle.length / 2
        half_width = self.config.vehicle.width / 2

        return ((waypoint.x, waypoint.y), (half_length * 2, half_width * 2))

    def _rectangles_collide(
        self,
        rect1: Tuple[Tuple[float, float], Tuple[float, float]],
        rect2: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> bool:
        """Check if two rectangles collide"""
        (x1, y1), (w1, h1) = rect1
        (x2, y2), (w2, h2) = rect2

        # Add safety margin
        margin = self.config.obstacle_clearance

        return not (x1 + w1/2 + margin < x2 - w2/2 or
                   x1 - w1/2 - margin > x2 + w2/2 or
                   y1 + h1/2 + margin < y2 - h2/2 or
                   y1 - h1/2 - margin > y2 + h2/2)

    def visualize_parking(
        self,
        image: np.ndarray,
        trajectory: Optional[ParkingTrajectory] = None,
        spots: Optional[List[ParkingSpot]] = None
    ) -> np.ndarray:
        """
        Visualize parking spots and trajectory

        Args:
            image: Input image or bird's eye view
            trajectory: Parking trajectory to visualize
            spots: Parking spots to visualize

        Returns:
            Visualization image
        """
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw parking spots
        if spots:
            for spot in spots:
                color = self._get_quality_color(spot.quality)

                # Draw spot corners
                corners_img = [(int(c[0] * 10 + vis_image.shape[1]/2),
                               int(vis_image.shape[0] - c[1] * 10))
                              for c in spot.corners]

                corners_array = np.array(corners_img, dtype=np.int32)
                cv2.polylines(vis_image, [corners_array], True, color, 2)

                # Draw spot label
                center_x = sum(c[0] for c in corners_img) // 4
                center_y = sum(c[1] for c in corners_img) // 4
                label = f"{spot.mode.value[:4].upper()} {spot.quality.value[:4]}"
                cv2.putText(vis_image, label, (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw trajectory
        if trajectory:
            for i, wp in enumerate(trajectory.waypoints):
                # Convert to image coordinates
                img_x = int(wp.x * 10 + vis_image.shape[1] / 2)
                img_y = int(vis_image.shape[0] - wp.y * 10)

                if 0 <= img_x < vis_image.shape[1] and 0 <= img_y < vis_image.shape[0]:
                    # Color based on gear
                    color = (0, 255, 0) if wp.gear == "D" else (0, 0, 255)
                    cv2.circle(vis_image, (img_x, img_y), 3, color, -1)

                    # Draw steering indicator
                    if abs(wp.steering_angle) > 5:
                        arrow_len = 15
                        angle = wp.heading + math.radians(wp.steering_angle)
                        end_x = int(img_x + arrow_len * math.cos(angle))
                        end_y = int(img_y - arrow_len * math.sin(angle))
                        cv2.arrowedLine(vis_image, (img_x, img_y), (end_x, end_y), color, 1)

                    # Connect waypoints
                    if i > 0:
                        prev_wp = trajectory.waypoints[i-1]
                        prev_x = int(prev_wp.x * 10 + vis_image.shape[1] / 2)
                        prev_y = int(vis_image.shape[0] - prev_wp.y * 10)
                        if 0 <= prev_x < vis_image.shape[1] and 0 <= prev_y < vis_image.shape[0]:
                            cv2.line(vis_image, (prev_x, prev_y), (img_x, img_y), color, 2)

        return vis_image

    def _get_quality_color(self, quality: SpotQuality) -> Tuple[int, int, int]:
        """Get color for spot quality"""
        color_map = {
            SpotQuality.EXCELLENT: (0, 255, 0),
            SpotQuality.GOOD: (0, 255, 255),
            SpotQuality.ACCEPTABLE: (0, 165, 255),
            SpotQuality.DIFFICULT: (0, 100, 255),
            SpotQuality.UNSUITABLE: (0, 0, 255)
        }
        return color_map.get(quality, (128, 128, 128))

    def get_statistics(self) -> Dict[str, Any]:
        """Get parking assist statistics"""
        success_rate = (self.successful_parking_plans / max(self.total_spots_detected, 1) * 100)

        return {
            'total_spots_detected': self.total_spots_detected,
            'successful_plans': self.successful_parking_plans,
            'planning_success_rate': success_rate,
            'current_phase': self.current_phase.value
        }
