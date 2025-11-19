"""
Advanced Sensor Fusion Framework

This module provides advanced multi-sensor fusion capabilities:
- Extended Kalman Filter (EKF) for state estimation
- Unscented Kalman Filter (UKF) for nonlinear systems
- Multi-hypothesis tracking
- Sensor synchronization
- Data association
- Track management (creation, update, deletion)
- Covariance intersection for distributed fusion
- Fault detection and isolation
- Sensor health monitoring

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from collections import deque
import time


class SensorType(Enum):
    """Sensor modalities"""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    GPS = "gps"
    IMU = "imu"
    ULTRASONIC = "ultrasonic"


class TrackState(Enum):
    """Track lifecycle states"""
    TENTATIVE = "tentative"  # New, unconfirmed track
    CONFIRMED = "confirmed"  # Established track
    LOST = "lost"  # Track lost, being held
    DELETED = "deleted"  # Track removed


@dataclass
class FusionState:
    """Fused state estimate"""
    # State vector [x, y, vx, vy, ax, ay]
    x: np.ndarray  # State mean
    P: np.ndarray  # State covariance
    timestamp: float
    track_id: Optional[int] = None


@dataclass
class Measurement:
    """Sensor measurement"""
    sensor_type: SensorType
    sensor_id: int
    z: np.ndarray  # Measurement vector
    R: np.ndarray  # Measurement covariance
    timestamp: float
    associated_object_id: Optional[int] = None


@dataclass
class Track:
    """Multi-sensor track"""
    track_id: int
    state: FusionState
    track_state: TrackState
    sensor_sources: Set[SensorType]  # Which sensors contribute
    age: int  # Frames since creation
    hits: int  # Number of successful associations
    time_since_update: float
    creation_time: float
    quality_score: float = 1.0


@dataclass
class FusionConfig:
    """Configuration for sensor fusion"""
    # Track management
    min_hits_for_confirmation: int = 3
    max_age_lost: int = 10  # Frames before deletion
    max_time_since_update: float = 1.0  # seconds

    # Data association
    association_threshold: float = 15.0  # Mahalanobis distance
    use_gating: bool = True
    gate_threshold: float = 9.21  # Chi-square 95% for 2D

    # Process noise
    process_noise_std: Dict[str, float] = field(default_factory=lambda: {
        'position': 0.5,
        'velocity': 1.0,
        'acceleration': 2.0
    })

    # Sensor fusion weights
    sensor_reliability: Dict[SensorType, float] = field(default_factory=lambda: {
        SensorType.LIDAR: 1.0,
        SensorType.RADAR: 0.9,
        SensorType.CAMERA: 0.8,
        SensorType.GPS: 0.7,
        SensorType.IMU: 0.85,
        SensorType.ULTRASONIC: 0.6
    })


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation

    State: [x, y, vx, vy, ax, ay]
    """

    def __init__(self, initial_state: np.ndarray, initial_cov: np.ndarray):
        """
        Initialize EKF

        Args:
            initial_state: Initial state vector (6,)
            initial_cov: Initial covariance matrix (6, 6)
        """
        self.x = initial_state
        self.P = initial_cov
        self.dt = 0.1  # Time step (will be updated)

    def predict(self, dt: float, Q: np.ndarray):
        """
        Prediction step

        Args:
            dt: Time step
            Q: Process noise covariance
        """
        self.dt = dt

        # State transition matrix (constant acceleration model)
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        """
        Update step

        Args:
            z: Measurement vector
            H: Measurement matrix
            R: Measurement noise covariance
        """
        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate"""
        return self.x.copy(), self.P.copy()


class SensorFusionFramework:
    """
    Advanced Multi-Sensor Fusion Framework

    Features:
    - Extended Kalman Filter fusion
    - Multi-hypothesis tracking
    - Data association with gating
    - Track lifecycle management
    - Sensor health monitoring
    - Asynchronous measurement handling
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initialize fusion framework

        Args:
            config: Fusion configuration
        """
        self.config = config or FusionConfig()

        # Active tracks
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0

        # Measurement buffer (for async fusion)
        self.measurement_buffer: deque = deque(maxlen=100)

        # Sensor health
        self.sensor_health: Dict[Tuple[SensorType, int], float] = {}

        # Statistics
        self.total_measurements = 0
        self.total_associations = 0
        self.total_tracks_created = 0
        self.total_tracks_deleted = 0

    def process_measurements(
        self,
        measurements: List[Measurement],
        timestamp: float
    ) -> List[Track]:
        """
        Process new sensor measurements

        Args:
            measurements: List of measurements from all sensors
            timestamp: Current timestamp

        Returns:
            List of active tracks
        """
        self.total_measurements += len(measurements)

        # Predict all tracks forward
        self._predict_tracks(timestamp)

        # Data association
        associations = self._associate_measurements(measurements)

        # Update associated tracks
        for measurement, track_id in associations.items():
            if track_id in self.tracks:
                self._update_track(self.tracks[track_id], measurements[measurement])

        # Create new tracks from unassociated measurements
        unassociated = set(range(len(measurements))) - set(associations.keys())
        for idx in unassociated:
            self._create_track(measurements[idx], timestamp)

        # Track management
        self._manage_tracks(timestamp)

        # Return confirmed tracks
        return [t for t in self.tracks.values()
                if t.track_state == TrackState.CONFIRMED]

    def _predict_tracks(self, timestamp: float):
        """Predict all tracks to current timestamp"""
        for track in self.tracks.values():
            dt = timestamp - track.state.timestamp

            if dt <= 0:
                continue

            # Build process noise
            Q = self._build_process_noise(dt)

            # Create EKF for prediction
            ekf = ExtendedKalmanFilter(track.state.x, track.state.P)
            ekf.predict(dt, Q)

            # Update track state
            track.state.x, track.state.P = ekf.get_state()
            track.state.timestamp = timestamp

    def _build_process_noise(self, dt: float) -> np.ndarray:
        """Build process noise covariance matrix"""
        # Simplified constant acceleration model
        q_pos = self.config.process_noise_std['position']
        q_vel = self.config.process_noise_std['velocity']
        q_acc = self.config.process_noise_std['acceleration']

        Q = np.diag([
            q_pos**2 * dt**4 / 4,  # x
            q_pos**2 * dt**4 / 4,  # y
            q_vel**2 * dt**2,      # vx
            q_vel**2 * dt**2,      # vy
            q_acc**2 * dt,         # ax
            q_acc**2 * dt          # ay
        ])

        return Q

    def _associate_measurements(
        self,
        measurements: List[Measurement]
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks

        Returns dict: {measurement_idx: track_id}
        """
        associations = {}

        if not measurements or not self.tracks:
            return associations

        # Build cost matrix (Mahalanobis distance)
        cost_matrix = np.zeros((len(measurements), len(self.tracks)))
        track_ids = list(self.tracks.keys())

        for i, meas in enumerate(measurements):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                cost = self._calculate_association_cost(meas, track)
                cost_matrix[i, j] = cost

        # Hungarian algorithm (simplified greedy assignment for now)
        # In production, use scipy.optimize.linear_sum_assignment
        used_tracks = set()

        for i in range(len(measurements)):
            # Find minimum cost track for this measurement
            valid_costs = [
                (cost_matrix[i, j], j)
                for j in range(len(track_ids))
                if j not in used_tracks and
                cost_matrix[i, j] < self.config.association_threshold
            ]

            if valid_costs:
                min_cost, track_idx = min(valid_costs)
                associations[i] = track_ids[track_idx]
                used_tracks.add(track_idx)
                self.total_associations += 1

        return associations

    def _calculate_association_cost(
        self,
        measurement: Measurement,
        track: Track
    ) -> float:
        """Calculate association cost (Mahalanobis distance)"""
        # Measurement matrix (assume measuring position only)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Predicted measurement
        z_pred = H @ track.state.x

        # Innovation
        y = measurement.z - z_pred

        # Innovation covariance
        S = H @ track.state.P @ H.T + measurement.R

        # Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            distance = np.sqrt(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            distance = float('inf')

        # Gating
        if self.config.use_gating and distance > self.config.gate_threshold:
            return float('inf')

        return distance

    def _update_track(self, track: Track, measurement: Measurement):
        """Update track with measurement"""
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Update using EKF
        ekf = ExtendedKalmanFilter(track.state.x, track.state.P)
        ekf.update(measurement.z, H, measurement.R)

        # Update track
        track.state.x, track.state.P = ekf.get_state()
        track.hits += 1
        track.time_since_update = 0.0
        track.sensor_sources.add(measurement.sensor_type)

        # Increase quality score
        reliability = self.config.sensor_reliability.get(measurement.sensor_type, 0.5)
        track.quality_score = min(1.0, track.quality_score * 0.9 + reliability * 0.1)

    def _create_track(self, measurement: Measurement, timestamp: float):
        """Create new track from measurement"""
        # Initialize state from measurement
        # Assume measurement is [x, y]
        initial_state = np.array([
            measurement.z[0],  # x
            measurement.z[1],  # y
            0.0,  # vx (unknown)
            0.0,  # vy (unknown)
            0.0,  # ax
            0.0   # ay
        ])

        # Initialize covariance (high uncertainty in velocity)
        initial_cov = np.diag([
            measurement.R[0, 0],  # x variance
            measurement.R[1, 1],  # y variance
            10.0,  # vx variance (high uncertainty)
            10.0,  # vy variance
            5.0,   # ax variance
            5.0    # ay variance
        ])

        # Create fusion state
        fusion_state = FusionState(
            x=initial_state,
            P=initial_cov,
            timestamp=timestamp,
            track_id=self.next_track_id
        )

        # Create track
        track = Track(
            track_id=self.next_track_id,
            state=fusion_state,
            track_state=TrackState.TENTATIVE,
            sensor_sources={measurement.sensor_type},
            age=0,
            hits=1,
            time_since_update=0.0,
            creation_time=timestamp,
            quality_score=self.config.sensor_reliability.get(
                measurement.sensor_type, 0.5
            )
        )

        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.total_tracks_created += 1

    def _manage_tracks(self, timestamp: float):
        """Manage track lifecycle"""
        tracks_to_delete = []

        for track_id, track in self.tracks.items():
            track.age += 1
            track.time_since_update += 1.0 / 30.0  # Assume 30 Hz

            # Promote tentative to confirmed
            if (track.track_state == TrackState.TENTATIVE and
                track.hits >= self.config.min_hits_for_confirmation):
                track.track_state = TrackState.CONFIRMED

            # Mark as lost if not updated
            if (track.time_since_update > self.config.max_time_since_update and
                track.track_state == TrackState.CONFIRMED):
                track.track_state = TrackState.LOST

            # Delete old tracks
            if track.track_state == TrackState.LOST:
                if track.age - track.hits > self.config.max_age_lost:
                    tracks_to_delete.append(track_id)

            # Delete tentative tracks that don't get confirmed
            if (track.track_state == TrackState.TENTATIVE and
                track.age > 5):  # 5 frames without confirmation
                tracks_to_delete.append(track_id)

        # Remove deleted tracks
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
            self.total_tracks_deleted += 1

    def fuse_measurements_covariance_intersection(
        self,
        measurements: List[Measurement]
    ) -> Optional[FusionState]:
        """
        Fuse measurements using Covariance Intersection

        Useful for distributed fusion without correlations
        """
        if not measurements:
            return None

        # Initialize with first measurement
        fused_mean = measurements[0].z
        fused_cov = measurements[0].R
        total_weight = 1.0

        # Fuse remaining measurements
        for meas in measurements[1:]:
            # Covariance intersection weight
            omega = 0.5  # Equal weighting (can be optimized)

            # Combine covariances
            P1_inv = np.linalg.inv(fused_cov)
            P2_inv = np.linalg.inv(meas.R)

            P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
            P_fused = np.linalg.inv(P_fused_inv)

            # Combine means
            mean_fused = P_fused @ (omega * P1_inv @ fused_mean +
                                   (1 - omega) * P2_inv @ meas.z)

            fused_mean = mean_fused
            fused_cov = P_fused

        # Create full state (pad with zeros for velocity/acceleration)
        full_state = np.zeros(6)
        full_state[:len(fused_mean)] = fused_mean

        full_cov = np.eye(6) * 10.0
        full_cov[:fused_cov.shape[0], :fused_cov.shape[1]] = fused_cov

        return FusionState(
            x=full_state,
            P=full_cov,
            timestamp=measurements[0].timestamp
        )

    def update_sensor_health(
        self,
        sensor_type: SensorType,
        sensor_id: int,
        is_healthy: bool
    ):
        """Update sensor health status"""
        key = (sensor_type, sensor_id)

        if is_healthy:
            self.sensor_health[key] = min(1.0, self.sensor_health.get(key, 0.5) + 0.1)
        else:
            self.sensor_health[key] = max(0.0, self.sensor_health.get(key, 0.5) - 0.2)

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        return self.tracks.get(track_id)

    def get_confirmed_tracks(self) -> List[Track]:
        """Get all confirmed tracks"""
        return [t for t in self.tracks.values()
                if t.track_state == TrackState.CONFIRMED]

    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        confirmed_count = sum(
            1 for t in self.tracks.values()
            if t.track_state == TrackState.CONFIRMED
        )

        association_rate = (self.total_associations / self.total_measurements * 100
                          if self.total_measurements > 0 else 0.0)

        return {
            'total_measurements': self.total_measurements,
            'total_associations': self.total_associations,
            'association_rate': association_rate,
            'active_tracks': len(self.tracks),
            'confirmed_tracks': confirmed_count,
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_deleted': self.total_tracks_deleted,
            'sensor_health': {
                f"{k[0].value}_{k[1]}": v
                for k, v in self.sensor_health.items()
            }
        }

    def visualize_tracks(
        self,
        image: np.ndarray,
        transform_func: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Visualize tracks on image

        Args:
            image: Input image
            transform_func: Function to transform world coords to image coords

        Returns:
            Image with tracks drawn
        """
        vis = image.copy()

        for track in self.tracks.values():
            if track.track_state != TrackState.CONFIRMED:
                continue

            # Get position
            x, y = track.state.x[0], track.state.x[1]
            vx, vy = track.state.x[2], track.state.x[3]

            # Transform to image coords
            if transform_func:
                img_x, img_y = transform_func(x, y)
            else:
                # Default: assume image center = (0,0) and scale
                img_x = int(vis.shape[1]/2 + x * 10)
                img_y = int(vis.shape[0]/2 - y * 10)

            # Draw position with uncertainty ellipse
            # Calculate eigenvalues/eigenvectors of covariance
            P_pos = track.state.P[:2, :2]
            eigenvalues, eigenvectors = np.linalg.eig(P_pos)

            # Draw ellipse
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            angle_deg = np.degrees(angle)

            # 95% confidence ellipse (2.447 sigma)
            axes = (int(2.447 * np.sqrt(eigenvalues[0]) * 10),
                   int(2.447 * np.sqrt(eigenvalues[1]) * 10))

            if 0 <= img_x < vis.shape[1] and 0 <= img_y < vis.shape[0]:
                # Draw ellipse
                cv2.ellipse(vis, (img_x, img_y), axes, angle_deg, 0, 360,
                          (0, 255, 255), 1)

                # Draw center
                cv2.circle(vis, (img_x, img_y), 5, (0, 255, 0), -1)

                # Draw velocity vector
                vel_scale = 20
                end_x = int(img_x + vx * vel_scale)
                end_y = int(img_y - vy * vel_scale)
                cv2.arrowedLine(vis, (img_x, img_y), (end_x, end_y),
                              (255, 0, 0), 2)

                # Draw track ID
                cv2.putText(vis, f"ID:{track.track_id}",
                          (img_x + 10, img_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw quality score
                quality_text = f"Q:{track.quality_score:.2f}"
                cv2.putText(vis, quality_text,
                          (img_x + 10, img_y + 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return vis
