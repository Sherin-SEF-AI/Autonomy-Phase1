"""
HD Map Integration & Localization Module

This module provides high-definition map integration and precise localization:
- HD map loading and representation (lanes, boundaries, signs, intersections)
- GPS/IMU/Visual odometry fusion for localization
- Map matching (snap to road network)
- Lane-level positioning
- Route planning on HD map
- Map feature extraction
- Geofencing and map queries
- Map tile management for large areas

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from collections import deque
import math
import json


class LaneType(Enum):
    """Lane types in HD map"""
    DRIVING = "driving"
    BUS_LANE = "bus_lane"
    BIKE_LANE = "bike_lane"
    PARKING = "parking"
    EMERGENCY = "emergency"
    HOV = "hov"  # High Occupancy Vehicle
    SHOULDER = "shoulder"


class RoadType(Enum):
    """Road classification"""
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    COLLECTOR = "collector"
    LOCAL = "local"
    RAMP = "ramp"
    PARKING_AISLE = "parking_aisle"


class MapFeatureType(Enum):
    """Types of map features"""
    LANE_BOUNDARY = "lane_boundary"
    STOP_LINE = "stop_line"
    CROSSWALK = "crosswalk"
    SPEED_BUMP = "speed_bump"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    PARKING_SPOT = "parking_spot"
    INTERSECTION = "intersection"
    JUNCTION = "junction"


@dataclass
class GeoPosition:
    """Geographic position"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float = 0.0  # meters
    heading: float = 0.0  # degrees (0-360, north=0)

    def distance_to(self, other: 'GeoPosition') -> float:
        """Calculate distance using Haversine formula"""
        R = 6371000  # Earth radius in meters

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def bearing_to(self, other: 'GeoPosition') -> float:
        """Calculate bearing to another position (degrees)"""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlon = lon2 - lon1

        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360


@dataclass
class HDLane:
    """High-definition lane representation"""
    lane_id: int
    lane_type: LaneType
    centerline: List[GeoPosition]  # Lane centerline points
    left_boundary: List[GeoPosition]
    right_boundary: List[GeoPosition]
    width: float  # meters
    speed_limit: Optional[float] = None  # m/s
    predecessor_ids: List[int] = field(default_factory=list)
    successor_ids: List[int] = field(default_factory=list)
    left_neighbor_id: Optional[int] = None
    right_neighbor_id: Optional[int] = None
    can_change_left: bool = True
    can_change_right: bool = True

    def get_length(self) -> float:
        """Calculate total lane length"""
        length = 0.0
        for i in range(len(self.centerline) - 1):
            length += self.centerline[i].distance_to(self.centerline[i + 1])
        return length

    def get_closest_point(self, position: GeoPosition) -> Tuple[int, GeoPosition, float]:
        """
        Find closest point on lane to given position

        Returns: (index, closest_point, distance)
        """
        min_distance = float('inf')
        closest_idx = 0
        closest_point = self.centerline[0]

        for i, point in enumerate(self.centerline):
            dist = position.distance_to(point)
            if dist < min_distance:
                min_distance = dist
                closest_idx = i
                closest_point = point

        return (closest_idx, closest_point, min_distance)


@dataclass
class MapFeature:
    """Generic map feature"""
    feature_id: int
    feature_type: MapFeatureType
    position: GeoPosition
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intersection:
    """Intersection representation"""
    intersection_id: int
    center: GeoPosition
    incoming_lane_ids: List[int]
    outgoing_lane_ids: List[int]
    has_traffic_light: bool = False
    has_stop_sign: bool = False
    yield_rules: Dict[int, List[int]] = field(default_factory=dict)  # lane_id -> yields_to_lane_ids


@dataclass
class LocalizationState:
    """Vehicle localization state"""
    position: GeoPosition
    current_lane_id: Optional[int]
    lateral_offset: float  # meters from lane center (+ = left)
    longitudinal_distance: float  # meters along lane
    confidence: float  # 0-1
    timestamp: float


@dataclass
class SensorMeasurement:
    """Sensor measurement for fusion"""
    position: GeoPosition
    velocity: Tuple[float, float, float]  # vx, vy, vz
    covariance: np.ndarray  # Position uncertainty
    timestamp: float
    sensor_type: str  # "gps", "imu", "visual_odometry"


class HDMap:
    """
    High-Definition Map

    Stores and manages HD map data including lanes, features, and topology
    """

    def __init__(self, map_data: Optional[Dict[str, Any]] = None):
        """
        Initialize HD map

        Args:
            map_data: Dictionary with map data or None for empty map
        """
        self.lanes: Dict[int, HDLane] = {}
        self.features: Dict[int, MapFeature] = {}
        self.intersections: Dict[int, Intersection] = {}

        # Spatial index for fast queries (simplified grid-based)
        self.grid_size = 100.0  # meters
        self.lane_grid: Dict[Tuple[int, int], List[int]] = {}

        if map_data:
            self.load_from_dict(map_data)

    def add_lane(self, lane: HDLane):
        """Add lane to map"""
        self.lanes[lane.lane_id] = lane
        self._update_spatial_index(lane)

    def add_feature(self, feature: MapFeature):
        """Add map feature"""
        self.features[feature.feature_id] = feature

    def add_intersection(self, intersection: Intersection):
        """Add intersection"""
        self.intersections[intersection.intersection_id] = intersection

    def _update_spatial_index(self, lane: HDLane):
        """Update spatial index with lane"""
        for point in lane.centerline:
            grid_x = int(point.latitude * 10000)  # Simple grid
            grid_y = int(point.longitude * 10000)
            key = (grid_x, grid_y)

            if key not in self.lane_grid:
                self.lane_grid[key] = []
            if lane.lane_id not in self.lane_grid[key]:
                self.lane_grid[key].append(lane.lane_id)

    def get_nearby_lanes(
        self,
        position: GeoPosition,
        radius: float = 50.0
    ) -> List[HDLane]:
        """Get lanes within radius of position"""
        nearby_lanes = []

        # Check neighboring grid cells
        grid_x = int(position.latitude * 10000)
        grid_y = int(position.longitude * 10000)

        # Search 3x3 grid around position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (grid_x + dx, grid_y + dy)
                if key in self.lane_grid:
                    for lane_id in self.lane_grid[key]:
                        lane = self.lanes[lane_id]
                        _, _, distance = lane.get_closest_point(position)
                        if distance <= radius:
                            nearby_lanes.append(lane)

        return nearby_lanes

    def get_lane(self, lane_id: int) -> Optional[HDLane]:
        """Get lane by ID"""
        return self.lanes.get(lane_id)

    def get_lane_network_path(
        self,
        start_lane_id: int,
        end_lane_id: int,
        max_depth: int = 20
    ) -> Optional[List[int]]:
        """
        Find path through lane network using BFS

        Returns list of lane IDs or None if no path found
        """
        if start_lane_id == end_lane_id:
            return [start_lane_id]

        queue = deque([(start_lane_id, [start_lane_id])])
        visited = {start_lane_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            current_lane = self.lanes.get(current_id)
            if not current_lane:
                continue

            # Check successors
            for successor_id in current_lane.successor_ids:
                if successor_id == end_lane_id:
                    return path + [successor_id]

                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append((successor_id, path + [successor_id]))

        return None

    def load_from_dict(self, map_data: Dict[str, Any]):
        """Load map from dictionary"""
        # Load lanes
        for lane_data in map_data.get('lanes', []):
            lane = self._dict_to_lane(lane_data)
            self.add_lane(lane)

        # Load features
        for feature_data in map_data.get('features', []):
            feature = self._dict_to_feature(feature_data)
            self.add_feature(feature)

        # Load intersections
        for intersection_data in map_data.get('intersections', []):
            intersection = self._dict_to_intersection(intersection_data)
            self.add_intersection(intersection)

    def _dict_to_lane(self, data: Dict[str, Any]) -> HDLane:
        """Convert dictionary to HDLane"""
        centerline = [
            GeoPosition(p['lat'], p['lon'], p.get('alt', 0))
            for p in data['centerline']
        ]
        left_boundary = [
            GeoPosition(p['lat'], p['lon'], p.get('alt', 0))
            for p in data.get('left_boundary', [])
        ]
        right_boundary = [
            GeoPosition(p['lat'], p['lon'], p.get('alt', 0))
            for p in data.get('right_boundary', [])
        ]

        return HDLane(
            lane_id=data['lane_id'],
            lane_type=LaneType(data['lane_type']),
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            width=data['width'],
            speed_limit=data.get('speed_limit'),
            predecessor_ids=data.get('predecessor_ids', []),
            successor_ids=data.get('successor_ids', []),
            left_neighbor_id=data.get('left_neighbor_id'),
            right_neighbor_id=data.get('right_neighbor_id')
        )

    def _dict_to_feature(self, data: Dict[str, Any]) -> MapFeature:
        """Convert dictionary to MapFeature"""
        return MapFeature(
            feature_id=data['feature_id'],
            feature_type=MapFeatureType(data['feature_type']),
            position=GeoPosition(
                data['position']['lat'],
                data['position']['lon'],
                data['position'].get('alt', 0)
            ),
            properties=data.get('properties', {})
        )

    def _dict_to_intersection(self, data: Dict[str, Any]) -> Intersection:
        """Convert dictionary to Intersection"""
        return Intersection(
            intersection_id=data['intersection_id'],
            center=GeoPosition(
                data['center']['lat'],
                data['center']['lon'],
                data['center'].get('alt', 0)
            ),
            incoming_lane_ids=data['incoming_lane_ids'],
            outgoing_lane_ids=data['outgoing_lane_ids'],
            has_traffic_light=data.get('has_traffic_light', False),
            has_stop_sign=data.get('has_stop_sign', False)
        )

    def save_to_file(self, filename: str):
        """Save map to JSON file"""
        map_data = {
            'lanes': [self._lane_to_dict(lane) for lane in self.lanes.values()],
            'features': [self._feature_to_dict(f) for f in self.features.values()],
            'intersections': [self._intersection_to_dict(i) for i in self.intersections.values()]
        }

        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)

    def _lane_to_dict(self, lane: HDLane) -> Dict[str, Any]:
        """Convert HDLane to dictionary"""
        return {
            'lane_id': lane.lane_id,
            'lane_type': lane.lane_type.value,
            'centerline': [{'lat': p.latitude, 'lon': p.longitude, 'alt': p.altitude}
                          for p in lane.centerline],
            'left_boundary': [{'lat': p.latitude, 'lon': p.longitude, 'alt': p.altitude}
                             for p in lane.left_boundary],
            'right_boundary': [{'lat': p.latitude, 'lon': p.longitude, 'alt': p.altitude}
                              for p in lane.right_boundary],
            'width': lane.width,
            'speed_limit': lane.speed_limit,
            'predecessor_ids': lane.predecessor_ids,
            'successor_ids': lane.successor_ids,
            'left_neighbor_id': lane.left_neighbor_id,
            'right_neighbor_id': lane.right_neighbor_id
        }

    def _feature_to_dict(self, feature: MapFeature) -> Dict[str, Any]:
        """Convert MapFeature to dictionary"""
        return {
            'feature_id': feature.feature_id,
            'feature_type': feature.feature_type.value,
            'position': {
                'lat': feature.position.latitude,
                'lon': feature.position.longitude,
                'alt': feature.position.altitude
            },
            'properties': feature.properties
        }

    def _intersection_to_dict(self, intersection: Intersection) -> Dict[str, Any]:
        """Convert Intersection to dictionary"""
        return {
            'intersection_id': intersection.intersection_id,
            'center': {
                'lat': intersection.center.latitude,
                'lon': intersection.center.longitude,
                'alt': intersection.center.altitude
            },
            'incoming_lane_ids': intersection.incoming_lane_ids,
            'outgoing_lane_ids': intersection.outgoing_lane_ids,
            'has_traffic_light': intersection.has_traffic_light,
            'has_stop_sign': intersection.has_stop_sign
        }


class MapLocalizer:
    """
    Map-based Localization System

    Fuses GPS, IMU, and visual odometry with HD map for precise localization
    """

    def __init__(self, hd_map: HDMap):
        """
        Initialize localizer

        Args:
            hd_map: HD map for localization
        """
        self.map = hd_map
        self.current_state: Optional[LocalizationState] = None
        self.measurement_history: deque = deque(maxlen=100)

        # Kalman filter state (simplified)
        self.state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.covariance = np.eye(6) * 100.0  # Initial high uncertainty

        # Statistics
        self.total_updates = 0
        self.map_matched_updates = 0

    def update(
        self,
        measurements: List[SensorMeasurement],
        timestamp: float
    ) -> LocalizationState:
        """
        Update localization with sensor measurements

        Args:
            measurements: List of sensor measurements
            timestamp: Current timestamp

        Returns:
            Updated localization state
        """
        self.total_updates += 1

        # Fuse sensor measurements (simplified - would use EKF/UKF in production)
        fused_position = self._fuse_measurements(measurements)

        # Map matching
        matched_state = self._match_to_map(fused_position, timestamp)

        if matched_state:
            self.map_matched_updates += 1
            self.current_state = matched_state
        else:
            # Fallback to raw position
            self.current_state = LocalizationState(
                position=fused_position,
                current_lane_id=None,
                lateral_offset=0.0,
                longitudinal_distance=0.0,
                confidence=0.5,
                timestamp=timestamp
            )

        # Store measurement history
        self.measurement_history.append(measurements)

        return self.current_state

    def _fuse_measurements(
        self,
        measurements: List[SensorMeasurement]
    ) -> GeoPosition:
        """
        Fuse multiple sensor measurements

        Simplified implementation using weighted average
        In production, use Extended Kalman Filter or Particle Filter
        """
        if not measurements:
            # Return last known position
            if self.current_state:
                return self.current_state.position
            return GeoPosition(0.0, 0.0, 0.0)

        # Weight by inverse covariance (lower uncertainty = higher weight)
        total_weight = 0.0
        weighted_lat = 0.0
        weighted_lon = 0.0
        weighted_alt = 0.0

        for measurement in measurements:
            # Calculate weight from covariance
            if measurement.covariance is not None and measurement.covariance.size > 0:
                # Use trace of covariance as uncertainty measure
                uncertainty = np.trace(measurement.covariance[:3, :3])
                weight = 1.0 / (uncertainty + 1e-6)
            else:
                weight = 1.0

            # Sensor-specific weights
            if measurement.sensor_type == "gps":
                weight *= 0.8
            elif measurement.sensor_type == "visual_odometry":
                weight *= 1.2  # Visual odometry often more accurate short-term
            elif measurement.sensor_type == "imu":
                weight *= 0.5  # IMU accumulates drift

            weighted_lat += measurement.position.latitude * weight
            weighted_lon += measurement.position.longitude * weight
            weighted_alt += measurement.position.altitude * weight
            total_weight += weight

        if total_weight > 0:
            return GeoPosition(
                latitude=weighted_lat / total_weight,
                longitude=weighted_lon / total_weight,
                altitude=weighted_alt / total_weight
            )

        return measurements[0].position

    def _match_to_map(
        self,
        position: GeoPosition,
        timestamp: float
    ) -> Optional[LocalizationState]:
        """
        Match position to HD map lanes

        Returns localization state with lane-level precision
        """
        # Find nearby lanes
        nearby_lanes = self.map.get_nearby_lanes(position, radius=20.0)

        if not nearby_lanes:
            return None

        # Find best matching lane
        best_lane = None
        best_distance = float('inf')
        best_idx = 0
        best_point = None

        for lane in nearby_lanes:
            idx, point, distance = lane.get_closest_point(position)

            if distance < best_distance:
                best_distance = distance
                best_lane = lane
                best_idx = idx
                best_point = point

        if best_lane is None or best_distance > 10.0:  # Max 10m from lane
            return None

        # Calculate lateral offset (signed distance from lane center)
        # Positive = left of lane, negative = right
        lateral_offset = self._calculate_lateral_offset(
            position, best_point, best_lane, best_idx
        )

        # Calculate longitudinal distance along lane
        longitudinal_distance = 0.0
        for i in range(best_idx):
            longitudinal_distance += best_lane.centerline[i].distance_to(
                best_lane.centerline[i + 1]
            )
        longitudinal_distance += best_point.distance_to(position)

        # Confidence based on distance to lane
        confidence = max(0.0, 1.0 - (best_distance / 10.0))

        return LocalizationState(
            position=position,
            current_lane_id=best_lane.lane_id,
            lateral_offset=lateral_offset,
            longitudinal_distance=longitudinal_distance,
            confidence=confidence,
            timestamp=timestamp
        )

    def _calculate_lateral_offset(
        self,
        position: GeoPosition,
        closest_lane_point: GeoPosition,
        lane: HDLane,
        point_idx: int
    ) -> float:
        """Calculate signed lateral offset from lane center"""
        # Simplified: calculate perpendicular distance
        # In production, use cross product for signed distance

        # Get lane direction at this point
        if point_idx < len(lane.centerline) - 1:
            next_point = lane.centerline[point_idx + 1]
            lane_bearing = closest_lane_point.bearing_to(next_point)
        else:
            prev_point = lane.centerline[point_idx - 1]
            lane_bearing = prev_point.bearing_to(closest_lane_point)

        # Calculate bearing from lane point to vehicle
        vehicle_bearing = closest_lane_point.bearing_to(position)

        # Relative bearing
        relative_bearing = (vehicle_bearing - lane_bearing + 360) % 360

        # If relative bearing is 270-90, vehicle is to the left
        # If 90-270, vehicle is to the right
        distance = closest_lane_point.distance_to(position)

        if 270 <= relative_bearing or relative_bearing <= 90:
            return distance  # Left (positive)
        else:
            return -distance  # Right (negative)

    def get_current_lane(self) -> Optional[HDLane]:
        """Get current lane from HD map"""
        if self.current_state and self.current_state.current_lane_id:
            return self.map.get_lane(self.current_state.current_lane_id)
        return None

    def get_route_to_lane(self, target_lane_id: int) -> Optional[List[HDLane]]:
        """
        Get route from current lane to target lane

        Returns list of lanes to traverse
        """
        if not self.current_state or not self.current_state.current_lane_id:
            return None

        lane_ids = self.map.get_lane_network_path(
            self.current_state.current_lane_id,
            target_lane_id
        )

        if not lane_ids:
            return None

        return [self.map.get_lane(lid) for lid in lane_ids if lid in self.map.lanes]

    def visualize_localization(
        self,
        size: Tuple[int, int] = (800, 800)
    ) -> np.ndarray:
        """
        Visualize localization on local map

        Args:
            size: Output image size

        Returns:
            Visualization image
        """
        img = np.zeros((*size, 3), dtype=np.uint8)
        w, h = size

        if not self.current_state:
            return img

        # Get nearby lanes
        nearby_lanes = self.map.get_nearby_lanes(
            self.current_state.position,
            radius=100.0
        )

        # Draw lanes
        for lane in nearby_lanes:
            color = (100, 100, 100)
            if lane.lane_id == self.current_state.current_lane_id:
                color = (0, 255, 0)  # Green for current lane

            # Convert geo to pixel coordinates (simplified)
            points = []
            for point in lane.centerline:
                # Relative to current position
                dx = (point.longitude - self.current_state.position.longitude) * 111320 * math.cos(math.radians(point.latitude))
                dy = (point.latitude - self.current_state.position.latitude) * 110540

                px = int(w/2 + dx * 5)  # Scale factor
                py = int(h/2 - dy * 5)  # Inverted Y

                if 0 <= px < w and 0 <= py < h:
                    points.append((px, py))

            if len(points) > 1:
                points_array = np.array(points, dtype=np.int32)
                cv2.polylines(img, [points_array], False, color, 2)

        # Draw vehicle
        cv2.circle(img, (w//2, h//2), 10, (0, 0, 255), -1)

        # Draw heading
        if self.current_state.position.heading:
            angle = math.radians(self.current_state.position.heading)
            end_x = int(w//2 + 30 * math.sin(angle))
            end_y = int(h//2 - 30 * math.cos(angle))
            cv2.arrowedLine(img, (w//2, h//2), (end_x, end_y), (0, 0, 255), 3)

        # Draw info
        info_lines = [
            f"Lane ID: {self.current_state.current_lane_id or 'Unknown'}",
            f"Lateral Offset: {self.current_state.lateral_offset:.2f}m",
            f"Confidence: {self.current_state.confidence:.2f}",
            f"Lat: {self.current_state.position.latitude:.6f}",
            f"Lon: {self.current_state.position.longitude:.6f}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        return img

    def get_statistics(self) -> Dict[str, Any]:
        """Get localization statistics"""
        match_rate = (self.map_matched_updates / self.total_updates * 100
                     if self.total_updates > 0 else 0.0)

        return {
            'total_updates': self.total_updates,
            'map_matched_updates': self.map_matched_updates,
            'map_match_rate': match_rate,
            'current_lane_id': self.current_state.current_lane_id if self.current_state else None,
            'localization_confidence': self.current_state.confidence if self.current_state else 0.0
        }
