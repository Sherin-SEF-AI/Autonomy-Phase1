"""
Trip analytics and reporting system.

Generates statistics, reports, and insights from perception data.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter, defaultdict

from utils.logger import get_logger


logger = get_logger()


@dataclass
class TripStatistics:
    """Complete trip statistics."""
    trip_id: str
    start_time: float
    end_time: Optional[float] = None

    # Duration
    duration_seconds: float = 0.0

    # Distance (requires GPS/odometry)
    distance_km: float = 0.0

    # Speed stats
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0

    # Detection stats
    total_detections: int = 0
    unique_objects_tracked: int = 0
    detections_by_class: Dict[str, int] = None

    # Scene stats
    scene_types_distribution: Dict[str, float] = None  # % in each scene type
    weather_distribution: Dict[str, float] = None  # % in each weather condition

    # Safety stats
    average_safety_score: float = 0.0
    min_safety_score: float = 100.0
    near_miss_count: int = 0
    lane_departure_count: int = 0
    collision_warning_count: int = 0

    # Performance stats
    avg_fps: float = 0.0
    avg_processing_time_ms: float = 0.0
    frames_processed: int = 0

    def __post_init__(self):
        if self.detections_by_class is None:
            self.detections_by_class = {}
        if self.scene_types_distribution is None:
            self.scene_types_distribution = {}
        if self.weather_distribution is None:
            self.weather_distribution = {}


class TripAnalyzer:
    """
    Analyzes trips and generates comprehensive reports.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize trip analyzer.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = output_dir or Path("data/analytics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Current trip data
        self.current_trip: Optional[TripStatistics] = None

        # Historical trips
        self.trip_history: List[TripStatistics] = []

        # Accumulation buffers for current trip
        self.detection_counter = Counter()
        self.scene_type_counter = Counter()
        self.weather_counter = Counter()
        self.safety_scores: List[float] = []
        self.speed_readings: List[float] = []
        self.fps_readings: List[float] = []
        self.processing_times: List[float] = []

        # Unique object tracking
        self.seen_track_ids = set()

        logger.info("Trip analyzer initialized")

    def start_trip(self, trip_id: Optional[str] = None) -> str:
        """
        Start a new trip.

        Args:
            trip_id: Optional trip ID

        Returns:
            Trip ID
        """
        if self.current_trip:
            logger.warning("Trip already in progress, finishing previous trip")
            self.end_trip()

        # Generate trip ID
        if not trip_id:
            trip_id = datetime.now().strftime("trip_%Y%m%d_%H%M%S")

        self.current_trip = TripStatistics(
            trip_id=trip_id,
            start_time=time.time()
        )

        # Reset buffers
        self.detection_counter.clear()
        self.scene_type_counter.clear()
        self.weather_counter.clear()
        self.safety_scores.clear()
        self.speed_readings.clear()
        self.fps_readings.clear()
        self.processing_times.clear()
        self.seen_track_ids.clear()

        logger.info(f"Started trip: {trip_id}")
        return trip_id

    def update_trip(
        self,
        detections: Optional[Dict[str, int]] = None,
        tracked_object_ids: Optional[List[int]] = None,
        scene_type: Optional[str] = None,
        weather: Optional[str] = None,
        safety_score: Optional[float] = None,
        speed_kmh: Optional[float] = None,
        fps: Optional[float] = None,
        processing_time_ms: Optional[float] = None,
        collision_warnings: int = 0,
        lane_departures: int = 0,
        near_misses: int = 0
    ):
        """
        Update current trip with new data.

        Args:
            detections: Dictionary of class_name -> count
            tracked_object_ids: List of currently tracked IDs
            scene_type: Current scene type
            weather: Current weather condition
            safety_score: Current safety score (0-100)
            speed_kmh: Current speed in km/h
            fps: Current FPS
            processing_time_ms: Frame processing time in ms
            collision_warnings: Number of collision warnings this frame
            lane_departures: Number of lane departures this frame
            near_misses: Number of near misses this frame
        """
        if not self.current_trip:
            logger.warning("No active trip, call start_trip() first")
            return

        # Update detection counts
        if detections:
            self.detection_counter.update(detections)
            self.current_trip.total_detections += sum(detections.values())

        # Track unique objects
        if tracked_object_ids:
            self.seen_track_ids.update(tracked_object_ids)

        # Update scene type
        if scene_type:
            self.scene_type_counter[scene_type] += 1

        # Update weather
        if weather:
            self.weather_counter[weather] += 1

        # Update safety score
        if safety_score is not None:
            self.safety_scores.append(safety_score)
            self.current_trip.min_safety_score = min(
                self.current_trip.min_safety_score,
                safety_score
            )

        # Update speed
        if speed_kmh is not None:
            self.speed_readings.append(speed_kmh)
            self.current_trip.max_speed_kmh = max(
                self.current_trip.max_speed_kmh,
                speed_kmh
            )

        # Update performance metrics
        if fps is not None:
            self.fps_readings.append(fps)

        if processing_time_ms is not None:
            self.processing_times.append(processing_time_ms)
            self.current_trip.frames_processed += 1

        # Update safety event counts
        self.current_trip.collision_warning_count += collision_warnings
        self.current_trip.lane_departure_count += lane_departures
        self.current_trip.near_miss_count += near_misses

    def end_trip(self) -> Optional[TripStatistics]:
        """
        End current trip and finalize statistics.

        Returns:
            Completed trip statistics
        """
        if not self.current_trip:
            logger.warning("No active trip to end")
            return None

        # Finalize trip
        self.current_trip.end_time = time.time()
        self.current_trip.duration_seconds = (
            self.current_trip.end_time - self.current_trip.start_time
        )

        # Calculate detection distribution
        self.current_trip.detections_by_class = dict(self.detection_counter)
        self.current_trip.unique_objects_tracked = len(self.seen_track_ids)

        # Calculate scene type distribution
        total_scene_samples = sum(self.scene_type_counter.values())
        if total_scene_samples > 0:
            self.current_trip.scene_types_distribution = {
                scene: (count / total_scene_samples) * 100
                for scene, count in self.scene_type_counter.items()
            }

        # Calculate weather distribution
        total_weather_samples = sum(self.weather_counter.values())
        if total_weather_samples > 0:
            self.current_trip.weather_distribution = {
                weather: (count / total_weather_samples) * 100
                for weather, count in self.weather_counter.items()
            }

        # Calculate averages
        if self.safety_scores:
            self.current_trip.average_safety_score = sum(self.safety_scores) / len(self.safety_scores)

        if self.speed_readings:
            self.current_trip.avg_speed_kmh = sum(self.speed_readings) / len(self.speed_readings)

        if self.fps_readings:
            self.current_trip.avg_fps = sum(self.fps_readings) / len(self.fps_readings)

        if self.processing_times:
            self.current_trip.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)

        # Save to history
        self.trip_history.append(self.current_trip)

        # Save report
        self.save_trip_report(self.current_trip)

        logger.info(f"Trip ended: {self.current_trip.trip_id} ({self.current_trip.duration_seconds:.1f}s)")

        completed_trip = self.current_trip
        self.current_trip = None

        return completed_trip

    def save_trip_report(self, trip: TripStatistics):
        """Save trip report to JSON file."""
        try:
            report_file = self.output_dir / f"{trip.trip_id}_report.json"

            with open(report_file, 'w') as f:
                json.dump(asdict(trip), f, indent=2)

            logger.info(f"Trip report saved to {report_file}")

        except Exception as e:
            logger.error(f"Failed to save trip report: {e}")

    def generate_summary_report(
        self,
        trips: Optional[List[TripStatistics]] = None
    ) -> Dict:
        """
        Generate summary report across multiple trips.

        Args:
            trips: List of trips to summarize (None = all history)

        Returns:
            Summary dictionary
        """
        if trips is None:
            trips = self.trip_history

        if not trips:
            return {"error": "No trips available"}

        # Aggregate statistics
        total_duration = sum(t.duration_seconds for t in trips)
        total_distance = sum(t.distance_km for t in trips)
        total_detections = sum(t.total_detections for t in trips)

        # Average metrics
        avg_safety = sum(t.average_safety_score for t in trips) / len(trips)
        avg_fps = sum(t.avg_fps for t in trips) / len(trips)

        # Safety events
        total_near_misses = sum(t.near_miss_count for t in trips)
        total_lane_departures = sum(t.lane_departure_count for t in trips)
        total_collision_warnings = sum(t.collision_warning_count for t in trips)

        # Most common detections
        all_detections = defaultdict(int)
        for trip in trips:
            for class_name, count in trip.detections_by_class.items():
                all_detections[class_name] += count

        top_detections = dict(
            sorted(all_detections.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        summary = {
            "report_generated": datetime.now().isoformat(),
            "trips_analyzed": len(trips),
            "total_duration_hours": total_duration / 3600.0,
            "total_distance_km": total_distance,
            "total_detections": total_detections,
            "average_safety_score": avg_safety,
            "average_fps": avg_fps,
            "total_near_misses": total_near_misses,
            "total_lane_departures": total_lane_departures,
            "total_collision_warnings": total_collision_warnings,
            "top_detected_classes": top_detections,
            "trips": [
                {
                    "trip_id": t.trip_id,
                    "duration_min": t.duration_seconds / 60.0,
                    "distance_km": t.distance_km,
                    "avg_safety_score": t.average_safety_score
                }
                for t in trips
            ]
        }

        return summary

    def save_summary_report(
        self,
        filename: str = "summary_report.json",
        trips: Optional[List[TripStatistics]] = None
    ):
        """Save summary report to file."""
        summary = self.generate_summary_report(trips)

        report_file = self.output_dir / filename

        try:
            with open(report_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary report saved to {report_file}")

        except Exception as e:
            logger.error(f"Failed to save summary report: {e}")

    def get_safety_trends(self) -> Dict:
        """Get safety score trends over trips."""
        if not self.trip_history:
            return {}

        scores = [t.average_safety_score for t in self.trip_history]
        timestamps = [t.start_time for t in self.trip_history]

        return {
            "trip_count": len(scores),
            "scores": scores,
            "timestamps": timestamps,
            "current_average": sum(scores) / len(scores) if scores else 0.0,
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
        }

    def get_statistics(self) -> Dict:
        """Get analyzer statistics."""
        return {
            "current_trip_id": self.current_trip.trip_id if self.current_trip else None,
            "current_trip_duration_min": (
                (time.time() - self.current_trip.start_time) / 60.0
                if self.current_trip else 0.0
            ),
            "total_trips": len(self.trip_history),
            "total_duration_hours": sum(t.duration_seconds for t in self.trip_history) / 3600.0,
            "reports_saved": len(list(self.output_dir.glob("*_report.json")))
        }
