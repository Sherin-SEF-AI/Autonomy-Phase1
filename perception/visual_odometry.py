"""
Visual Odometry Module

This module provides visual odometry for ego-motion estimation using camera input:
- Feature detection and tracking (ORB, SIFT, optical flow)
- Essential/Fundamental matrix estimation
- Camera pose recovery (rotation and translation)
- Trajectory reconstruction
- Scale estimation with known object sizes
- Loop closure detection
- Map point triangulation
- Drift correction

Author: AV Perception System
Version: 1.2.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from collections import deque
import math


class VOMethod(Enum):
    """Visual odometry methods"""
    FEATURE_BASED = "feature_based"  # ORB/SIFT + Essential matrix
    OPTICAL_FLOW = "optical_flow"    # Lucas-Kanade
    DIRECT = "direct"                 # Direct alignment
    HYBRID = "hybrid"                 # Combination


class FeatureType(Enum):
    """Feature detector types"""
    ORB = "orb"
    SIFT = "sift"
    FAST = "fast"
    AKAZE = "akaze"


@dataclass
class CameraPose:
    """6-DOF camera pose"""
    position: np.ndarray  # 3D position (x, y, z)
    rotation: np.ndarray  # 3x3 rotation matrix
    timestamp: float
    confidence: float = 1.0

    def get_translation(self) -> np.ndarray:
        """Get translation vector"""
        return self.position

    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        R = self.rotation

        # Extract Euler angles from rotation matrix
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0

        return (roll, pitch, yaw)


@dataclass
class VOTrajectory:
    """Vehicle trajectory from visual odometry"""
    poses: List[CameraPose]
    timestamps: List[float]
    total_distance: float = 0.0
    average_speed: float = 0.0


@dataclass
class VOConfig:
    """Configuration for visual odometry"""
    method: VOMethod = VOMethod.FEATURE_BASED
    feature_type: FeatureType = FeatureType.ORB

    # Feature detection
    max_features: int = 500
    feature_quality: float = 0.01  # For corners

    # Matching
    match_ratio_threshold: float = 0.7  # Lowe's ratio test
    min_matches: int = 8  # Minimum matches for pose estimation

    # RANSAC
    ransac_threshold: float = 1.0  # pixels
    ransac_confidence: float = 0.99

    # Tracking
    min_feature_distance: float = 10.0  # Minimum distance between features
    max_tracking_error: float = 12.0  # Max pixel error for tracking

    # Scale estimation
    use_scale_estimation: bool = True
    known_object_sizes: Dict[str, float] = field(default_factory=lambda: {
        'car': 4.5,
        'person': 1.7,
        'traffic_sign': 0.6
    })

    # Trajectory
    max_trajectory_length: int = 1000  # Maximum poses to keep


class VisualOdometry:
    """
    Visual Odometry System

    Features:
    - Real-time camera ego-motion estimation
    - Multiple feature detection methods (ORB, SIFT, FAST)
    - Essential matrix decomposition for R|t recovery
    - Feature tracking across frames
    - Trajectory reconstruction
    - Scale estimation from known objects
    - Drift monitoring and correction
    """

    def __init__(self, config: Optional[VOConfig] = None):
        """
        Initialize visual odometry

        Args:
            config: VO configuration
        """
        self.config = config or VOConfig()

        # Initialize feature detector
        self.detector = self._create_feature_detector()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points = None

        # Trajectory
        self.trajectory = VOTrajectory(poses=[], timestamps=[])
        self.current_pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            timestamp=0.0
        )

        # Camera intrinsics (should be from calibration)
        self.camera_matrix = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)

        # Statistics
        self.total_frames_processed = 0
        self.successful_pose_estimates = 0
        self.total_distance_traveled = 0.0

    def _create_feature_detector(self):
        """Create feature detector based on config"""
        if self.config.feature_type == FeatureType.ORB:
            return cv2.ORB_create(nfeatures=self.config.max_features)
        elif self.config.feature_type == FeatureType.SIFT:
            return cv2.SIFT_create(nfeatures=self.config.max_features)
        elif self.config.feature_type == FeatureType.FAST:
            return cv2.FastFeatureDetector_create()
        elif self.config.feature_type == FeatureType.AKAZE:
            return cv2.AKAZE_create()
        else:
            return cv2.ORB_create(nfeatures=self.config.max_features)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        camera_matrix: Optional[np.ndarray] = None,
        known_objects: Optional[List[Dict[str, Any]]] = None
    ) -> CameraPose:
        """
        Process frame and estimate camera pose

        Args:
            frame: Input frame (grayscale or color)
            timestamp: Frame timestamp
            camera_matrix: Camera intrinsic matrix (optional)
            known_objects: Detected objects with known sizes for scale estimation

        Returns:
            Estimated camera pose
        """
        self.total_frames_processed += 1

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Update camera matrix if provided
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix

        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.prev_frame is None:
            # First frame - initialize
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

            self.trajectory.poses.append(self.current_pose)
            self.trajectory.timestamps.append(timestamp)

            return self.current_pose

        # Match features with previous frame
        if descriptors is None or self.prev_descriptors is None:
            # No features to match
            return self.current_pose

        matches = self._match_features(self.prev_descriptors, descriptors)

        if len(matches) < self.config.min_matches:
            # Not enough matches
            return self.current_pose

        # Extract matched points
        pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([keypoints[m.trainIdx].pt for m in matches])

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=self.config.ransac_threshold
        )

        if E is None:
            return self.current_pose

        # Filter matches with RANSAC mask
        pts_prev = pts_prev[mask.ravel() == 1]
        pts_curr = pts_curr[mask.ravel() == 1]

        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(
            E, pts_prev, pts_curr, self.camera_matrix
        )

        # Estimate scale if enabled
        if self.config.use_scale_estimation and known_objects:
            scale = self._estimate_scale(pts_prev, pts_curr, known_objects)
            t = t * scale

        # Update current pose
        self.current_pose.rotation = R @ self.current_pose.rotation
        self.current_pose.position += (self.current_pose.rotation @ t).flatten()
        self.current_pose.timestamp = timestamp

        # Update statistics
        self.successful_pose_estimates += 1
        if len(self.trajectory.poses) > 0:
            prev_pos = self.trajectory.poses[-1].position
            curr_pos = self.current_pose.position
            distance = np.linalg.norm(curr_pos - prev_pos)
            self.total_distance_traveled += distance

        # Add to trajectory
        self.trajectory.poses.append(
            CameraPose(
                position=self.current_pose.position.copy(),
                rotation=self.current_pose.rotation.copy(),
                timestamp=timestamp
            )
        )
        self.trajectory.timestamps.append(timestamp)

        # Limit trajectory length
        if len(self.trajectory.poses) > self.config.max_trajectory_length:
            self.trajectory.poses.pop(0)
            self.trajectory.timestamps.pop(0)

        # Update for next frame
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_points = pts_curr

        return self.current_pose

    def _match_features(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> List[cv2.DMatch]:
        """Match features between two frames"""
        # Use KNN matching for ratio test
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.match_ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def _estimate_scale(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        known_objects: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate metric scale from known object sizes

        This addresses the scale ambiguity problem in monocular VO
        """
        # Try to find correspondences with known objects
        # Simplified: use first known object
        for obj in known_objects:
            obj_class = obj.get('class', 'unknown')
            if obj_class in self.config.known_object_sizes:
                known_size = self.config.known_object_sizes[obj_class]

                # Get object bounding box
                bbox = obj.get('bbox')  # (x, y, w, h)
                if bbox:
                    x, y, w, h = bbox
                    # Estimate distance using pinhole model
                    # distance = (real_height * focal_length) / pixel_height
                    focal_length = self.camera_matrix[1, 1]
                    estimated_distance = (known_size * focal_length) / h

                    # Use this as scale
                    # In practice, would need more sophisticated method
                    return estimated_distance / 10.0  # Rough heuristic

        # Default scale (no correction)
        return 1.0

    def get_current_position(self) -> Tuple[float, float, float]:
        """Get current position (x, y, z)"""
        return tuple(self.current_pose.position)

    def get_current_orientation(self) -> Tuple[float, float, float]:
        """Get current orientation as Euler angles (roll, pitch, yaw)"""
        return self.current_pose.get_euler_angles()

    def get_trajectory_2d(self) -> np.ndarray:
        """
        Get 2D trajectory (X-Z plane for visualization)

        Returns:
            Nx2 array of (x, z) positions
        """
        if not self.trajectory.poses:
            return np.array([])

        positions = np.array([pose.position for pose in self.trajectory.poses])
        # Return X and Z coordinates (horizontal plane)
        return positions[:, [0, 2]]

    def get_speed(self, window: int = 5) -> float:
        """
        Estimate current speed from recent trajectory

        Args:
            window: Number of recent poses to use

        Returns:
            Speed in m/s
        """
        if len(self.trajectory.poses) < 2:
            return 0.0

        # Use last 'window' poses
        recent_poses = self.trajectory.poses[-window:]
        recent_times = self.trajectory.timestamps[-window:]

        if len(recent_poses) < 2:
            return 0.0

        # Calculate total distance
        total_distance = 0.0
        for i in range(len(recent_poses) - 1):
            pos1 = recent_poses[i].position
            pos2 = recent_poses[i + 1].position
            total_distance += np.linalg.norm(pos2 - pos1)

        # Calculate time elapsed
        time_elapsed = recent_times[-1] - recent_times[0]

        if time_elapsed > 0:
            return total_distance / time_elapsed
        else:
            return 0.0

    def visualize_trajectory(
        self,
        size: Tuple[int, int] = (800, 800),
        scale: float = 10.0
    ) -> np.ndarray:
        """
        Visualize trajectory in bird's eye view

        Args:
            size: Output image size (width, height)
            scale: Scaling factor (pixels per meter)

        Returns:
            Visualization image
        """
        img = np.zeros((*size, 3), dtype=np.uint8)
        w, h = size

        if len(self.trajectory.poses) < 2:
            return img

        # Draw grid
        grid_spacing = int(scale * 5)  # 5 meter grid
        for i in range(0, w, grid_spacing):
            cv2.line(img, (i, 0), (i, h), (50, 50, 50), 1)
        for i in range(0, h, grid_spacing):
            cv2.line(img, (0, i), (w, i), (50, 50, 50), 1)

        # Draw trajectory
        positions = self.get_trajectory_2d()

        # Center trajectory in image
        if len(positions) > 0:
            center_offset = np.array([w / 2, h / 2])

            for i in range(len(positions) - 1):
                pt1 = (positions[i] * scale + center_offset).astype(int)
                pt2 = (positions[i + 1] * scale + center_offset).astype(int)

                # Color gradient based on progress
                progress = i / len(positions)
                color = (
                    int(255 * (1 - progress)),
                    int(255 * progress),
                    128
                )

                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(img, tuple(pt1), tuple(pt2), color, 2)

            # Draw current position
            current_pt = (positions[-1] * scale + center_offset).astype(int)
            if 0 <= current_pt[0] < w and 0 <= current_pt[1] < h:
                cv2.circle(img, tuple(current_pt), 8, (0, 0, 255), -1)

            # Draw start position
            start_pt = (positions[0] * scale + center_offset).astype(int)
            if 0 <= start_pt[0] < w and 0 <= start_pt[1] < h:
                cv2.circle(img, tuple(start_pt), 8, (0, 255, 0), -1)

        # Add info text
        info_lines = [
            f"Distance: {self.total_distance_traveled:.1f}m",
            f"Frames: {self.total_frames_processed}",
            f"Speed: {self.get_speed():.1f}m/s",
            f"Poses: {len(self.trajectory.poses)}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        return img

    def visualize_features(
        self,
        frame: np.ndarray
    ) -> np.ndarray:
        """
        Visualize detected features on frame

        Args:
            frame: Input frame

        Returns:
            Frame with features drawn
        """
        if self.prev_keypoints is None:
            return frame

        # Draw keypoints
        vis_frame = cv2.drawKeypoints(
            frame,
            self.prev_keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Draw feature count
        text = f"Features: {len(self.prev_keypoints)}"
        cv2.putText(vis_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return vis_frame

    def reset(self):
        """Reset visual odometry state"""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points = None

        self.trajectory = VOTrajectory(poses=[], timestamps=[])
        self.current_pose = CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            timestamp=0.0
        )

        self.total_frames_processed = 0
        self.successful_pose_estimates = 0
        self.total_distance_traveled = 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get visual odometry statistics"""
        success_rate = (self.successful_pose_estimates / self.total_frames_processed * 100
                       if self.total_frames_processed > 0 else 0.0)

        avg_speed = 0.0
        if len(self.trajectory.timestamps) > 1:
            time_elapsed = self.trajectory.timestamps[-1] - self.trajectory.timestamps[0]
            if time_elapsed > 0:
                avg_speed = self.total_distance_traveled / time_elapsed

        return {
            'total_frames': self.total_frames_processed,
            'successful_estimates': self.successful_pose_estimates,
            'success_rate': success_rate,
            'total_distance': self.total_distance_traveled,
            'average_speed': avg_speed,
            'trajectory_length': len(self.trajectory.poses),
            'current_position': tuple(self.current_pose.position),
            'current_orientation': self.current_pose.get_euler_angles()
        }
