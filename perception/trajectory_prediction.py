"""
Advanced trajectory prediction and path forecasting.

Predicts future positions of tracked objects using physics-based models
and machine learning techniques.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import time

from utils.data_structures import TrackedObject
from utils.logger import get_logger


logger = get_logger()


@dataclass
class PredictedTrajectory:
    """Predicted future trajectory for an object."""
    track_id: int
    current_position: Tuple[float, float, float]
    predicted_positions: List[Tuple[float, float, float]]  # Future positions at time steps
    time_steps: List[float]  # Time stamps for predictions (seconds ahead)
    confidence: float  # Prediction confidence (0-1)
    collision_risk: float  # Risk of collision (0-1)


class TrajectoryPredictor:
    """
    Predicts future trajectories of tracked objects.

    Uses multiple models:
    - Linear extrapolation (constant velocity)
    - Polynomial fitting (acceleration)
    - Kalman filtering (optimal estimation)
    """

    def __init__(
        self,
        prediction_horizon: float = 3.0,  # Seconds ahead
        prediction_steps: int = 10,  # Number of prediction points
        min_history: int = 5  # Minimum trajectory points needed
    ):
        """
        Initialize trajectory predictor.

        Args:
            prediction_horizon: How far ahead to predict (seconds)
            prediction_steps: Number of discrete prediction points
            min_history: Minimum trajectory history needed for prediction
        """
        self.prediction_horizon = prediction_horizon
        self.prediction_steps = prediction_steps
        self.min_history = min_history

        # Trajectory history per track
        self.trajectory_history: Dict[int, deque] = {}
        self.max_history = 20  # Keep last 20 positions

        logger.info(f"Trajectory predictor initialized: {prediction_horizon}s horizon, {prediction_steps} steps")

    def update_trajectory(self, tracked_object: TrackedObject):
        """
        Update trajectory history for an object.

        Args:
            tracked_object: Tracked object with current position
        """
        track_id = tracked_object.track_id

        if track_id not in self.trajectory_history:
            self.trajectory_history[track_id] = deque(maxlen=self.max_history)

        # Add current position with timestamp
        if tracked_object.current_position:
            self.trajectory_history[track_id].append({
                'timestamp': time.time(),
                'position': tracked_object.current_position,
                'velocity': tracked_object.current_velocity
            })

    def predict(self, track_id: int) -> Optional[PredictedTrajectory]:
        """
        Predict future trajectory for an object.

        Args:
            track_id: ID of tracked object

        Returns:
            PredictedTrajectory if prediction possible, None otherwise
        """
        if track_id not in self.trajectory_history:
            return None

        history = list(self.trajectory_history[track_id])

        if len(history) < self.min_history:
            return None

        # Extract position and time data
        timestamps = np.array([h['timestamp'] for h in history])
        positions = np.array([h['position'] for h in history])

        # Normalize time to start from 0
        times = timestamps - timestamps[0]

        # Current position
        current_pos = positions[-1]

        # Try multiple prediction methods and ensemble them
        linear_pred = self._predict_linear(times, positions)
        poly_pred = self._predict_polynomial(times, positions)

        # Ensemble predictions (weighted average)
        if linear_pred is not None and poly_pred is not None:
            # Weight recent velocity more if object is accelerating
            alpha = 0.6  # Weight for linear model
            predicted_positions = alpha * linear_pred + (1 - alpha) * poly_pred
            confidence = 0.8
        elif linear_pred is not None:
            predicted_positions = linear_pred
            confidence = 0.6
        elif poly_pred is not None:
            predicted_positions = poly_pred
            confidence = 0.5
        else:
            return None

        # Calculate time steps
        dt = self.prediction_horizon / self.prediction_steps
        time_steps = [dt * (i + 1) for i in range(self.prediction_steps)]

        # Calculate collision risk (simple distance-based for now)
        collision_risk = self._calculate_collision_risk(current_pos, predicted_positions)

        return PredictedTrajectory(
            track_id=track_id,
            current_position=tuple(current_pos),
            predicted_positions=[tuple(p) for p in predicted_positions],
            time_steps=time_steps,
            confidence=confidence,
            collision_risk=collision_risk
        )

    def _predict_linear(
        self,
        times: np.ndarray,
        positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Predict trajectory using linear extrapolation (constant velocity).

        Args:
            times: Time points
            positions: Position points (Nx3)

        Returns:
            Predicted positions or None
        """
        if len(times) < 2:
            return None

        try:
            # Fit linear model for each dimension
            predictions = []

            for i in range(self.prediction_steps):
                dt = (i + 1) * (self.prediction_horizon / self.prediction_steps)
                future_time = times[-1] + dt

                # Simple linear extrapolation based on recent velocity
                if len(times) >= 2:
                    # Calculate velocity from last two points
                    velocity = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
                    predicted_pos = positions[-1] + velocity * dt
                    predictions.append(predicted_pos)

            return np.array(predictions)

        except Exception as e:
            logger.debug(f"Linear prediction failed: {e}")
            return None

    def _predict_polynomial(
        self,
        times: np.ndarray,
        positions: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Predict trajectory using polynomial fitting (handles acceleration).

        Args:
            times: Time points
            positions: Position points (Nx3)

        Returns:
            Predicted positions or None
        """
        if len(times) < 3:
            return None

        try:
            # Fit 2nd degree polynomial for each dimension
            predictions = []

            for i in range(self.prediction_steps):
                dt = (i + 1) * (self.prediction_horizon / self.prediction_steps)
                future_time = times[-1] + dt

                # Fit polynomial for each coordinate
                pred_pos = np.zeros(3)
                for dim in range(3):
                    # Fit 2nd degree polynomial
                    coeffs = np.polyfit(times, positions[:, dim], deg=2)
                    poly = np.poly1d(coeffs)
                    pred_pos[dim] = poly(future_time)

                predictions.append(pred_pos)

            return np.array(predictions)

        except Exception as e:
            logger.debug(f"Polynomial prediction failed: {e}")
            return None

    def _calculate_collision_risk(
        self,
        current_pos: np.ndarray,
        predicted_positions: np.ndarray
    ) -> float:
        """
        Calculate collision risk based on predicted trajectory.

        Args:
            current_pos: Current position
            predicted_positions: Array of predicted positions

        Returns:
            Collision risk score (0-1)
        """
        # Simple risk based on proximity to ego vehicle (0, 0, 0)
        ego_pos = np.array([0.0, 0.0, 0.0])

        # Find minimum distance to ego in predicted trajectory
        distances = np.linalg.norm(predicted_positions - ego_pos, axis=1)
        min_distance = np.min(distances)

        # Risk increases as object gets closer
        if min_distance < 2.0:  # Critical: < 2m
            risk = 1.0
        elif min_distance < 5.0:  # High: < 5m
            risk = 0.8 - (min_distance - 2.0) / 3.0 * 0.3
        elif min_distance < 10.0:  # Medium: < 10m
            risk = 0.5 - (min_distance - 5.0) / 5.0 * 0.3
        else:  # Low: > 10m
            risk = max(0.0, 0.2 - (min_distance - 10.0) / 20.0 * 0.2)

        return risk

    def predict_all(
        self,
        tracked_objects: List[TrackedObject]
    ) -> Dict[int, PredictedTrajectory]:
        """
        Predict trajectories for all tracked objects.

        Args:
            tracked_objects: List of tracked objects

        Returns:
            Dictionary mapping track_id to predicted trajectory
        """
        # Update trajectories
        for obj in tracked_objects:
            self.update_trajectory(obj)

        # Predict for all objects
        predictions = {}
        for track_id in self.trajectory_history.keys():
            prediction = self.predict(track_id)
            if prediction:
                predictions[track_id] = prediction

        return predictions

    def get_collision_pairs(
        self,
        predictions: Dict[int, PredictedTrajectory],
        distance_threshold: float = 3.0
    ) -> List[Tuple[int, int, float]]:
        """
        Detect potential collisions between predicted trajectories.

        Args:
            predictions: Dictionary of predicted trajectories
            distance_threshold: Distance threshold for collision (meters)

        Returns:
            List of (track_id1, track_id2, min_distance) tuples
        """
        collision_pairs = []
        track_ids = list(predictions.keys())

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track_id1 = track_ids[i]
                track_id2 = track_ids[j]

                pred1 = predictions[track_id1]
                pred2 = predictions[track_id2]

                # Check if trajectories intersect
                positions1 = np.array(pred1.predicted_positions)
                positions2 = np.array(pred2.predicted_positions)

                # Calculate pairwise distances at each time step
                min_dist = float('inf')
                for p1, p2 in zip(positions1, positions2):
                    dist = np.linalg.norm(np.array(p1) - np.array(p2))
                    min_dist = min(min_dist, dist)

                if min_dist < distance_threshold:
                    collision_pairs.append((track_id1, track_id2, min_dist))

        return collision_pairs

    def clear_history(self, track_id: int):
        """
        Clear trajectory history for a track.

        Args:
            track_id: Track ID to clear
        """
        if track_id in self.trajectory_history:
            del self.trajectory_history[track_id]

    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove trajectory history for tracks that no longer exist.

        Args:
            active_track_ids: List of currently active track IDs
        """
        active_set = set(active_track_ids)
        to_remove = [tid for tid in self.trajectory_history.keys() if tid not in active_set]

        for track_id in to_remove:
            del self.trajectory_history[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old trajectory histories")
