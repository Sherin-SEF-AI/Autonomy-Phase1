#!/usr/bin/env python3
"""
Demonstration of advanced ADAS features.

Shows how to integrate and use:
- Multi-camera recording
- Traffic light detection
- Depth estimation
- Semantic segmentation
- Scene recognition
- Safety scoring
- Object re-identification
- Analytics & reporting
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Import new advanced modules
from recording.video_recorder import MultiCameraRecorder, RecordingConfig, RecordingEvent
from perception.traffic_light_detection import TrafficLightDetector
from perception.depth_estimation import DepthEstimator, DepthModel
from perception.semantic_segmentation import SemanticSegmenter
from perception.scene_recognition import SceneRecognizer
from perception.object_reidentification import ObjectReIdentifier, CrossCameraTracker
from safety.safety_scorer import SafetyScorer
from analytics.trip_analyzer import TripAnalyzer

# Import existing modules
from utils.logger import get_logger
from utils.data_structures import TrackedObject, LaneDetectionResult, SafetyWarning, WarningLevel


logger = get_logger()


class AdvancedADASDemo:
    """
    Demonstrates integration of all advanced ADAS features.
    """

    def __init__(self):
        """Initialize all advanced modules."""
        logger.info("Initializing Advanced ADAS Demo...")

        # Recording system
        self.recorder = MultiCameraRecorder(
            RecordingConfig(
                output_dir=Path("data/recordings"),
                codec="mp4v",
                fps=30,
                buffer_duration_sec=300.0,  # 5 minutes
                event_pre_buffer_sec=10.0,
                event_post_buffer_sec=10.0
            )
        )

        # Traffic light detection
        self.traffic_detector = TrafficLightDetector(
            enable_detection=True
        )

        # Depth estimation (fallback to simple if no torch)
        self.depth_estimator = DepthEstimator(
            model_type=DepthModel.SIMPLE  # Or MIDAS_SMALL if torch available
        )

        # Semantic segmentation (fallback to simple if no torch)
        self.segmenter = SemanticSegmenter(
            model_type="simple"  # Or "deeplabv3" if torch available
        )

        # Scene recognition
        self.scene_recognizer = SceneRecognizer(
            use_temporal_smoothing=True
        )

        # Re-identification
        self.reidentifier = ObjectReIdentifier(
            feature_type="histogram",
            similarity_threshold=0.7
        )
        self.cross_camera_tracker = CrossCameraTracker(self.reidentifier)

        # Safety scorer
        self.safety_scorer = SafetyScorer()

        # Trip analyzer
        self.trip_analyzer = TripAnalyzer(
            output_dir=Path("data/analytics")
        )

        logger.info("✓ All modules initialized successfully")

    def process_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        tracked_objects: list,
        lane_result: LaneDetectionResult = None
    ):
        """
        Process a single frame through all advanced features.

        Args:
            camera_id: Camera ID
            frame: BGR image
            tracked_objects: List of tracked objects
            lane_result: Optional lane detection result
        """
        timestamp = time.time()

        # 1. Add frame to recorder
        self.recorder.add_frame(camera_id, frame, timestamp)

        # 2. Detect traffic lights
        traffic_lights = self.traffic_detector.detect(frame)

        if traffic_lights:
            for light in traffic_lights:
                logger.debug(
                    f"Camera {camera_id}: Traffic light {light.state.value} "
                    f"at {light.position} (conf: {light.confidence:.2f})"
                )

            # Check for red light
            if self.traffic_detector.is_red_light(traffic_lights):
                logger.warning("🚦 RED LIGHT DETECTED - DO NOT PROCEED!")

        # 3. Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Get depth for tracked objects
        for obj in tracked_objects:
            if hasattr(obj, 'current_bbox'):
                depth_stats = self.depth_estimator.get_depth_for_bbox(
                    depth_map,
                    obj.current_bbox
                )
                # Could update object with depth info
                logger.debug(
                    f"Object {obj.track_id} depth: {depth_stats['mean']:.1f}"
                )

        # 4. Semantic segmentation
        segmentation_mask = self.segmenter.segment(frame)

        # Extract drivable area
        drivable_area = self.segmenter.extract_drivable_area(segmentation_mask)

        # Get class percentages
        class_percentages = self.segmenter.get_class_percentages(segmentation_mask)
        road_percentage = class_percentages.get('road', 0)

        # 5. Scene recognition
        scene_context = self.scene_recognizer.recognize(frame)

        logger.debug(
            f"Scene: {scene_context.weather.value}, "
            f"{scene_context.road_type.value}, "
            f"{scene_context.time_of_day.value}, "
            f"visibility: {scene_context.visibility_score:.2f}"
        )

        # Check for poor visibility
        if self.scene_recognizer.is_poor_visibility(scene_context):
            logger.warning("⚠ Poor visibility conditions detected!")

        # 6. Update re-identification
        for obj in tracked_objects:
            self.reidentifier.update_appearance(obj, frame, camera_id)

        # 7. Calculate safety score
        safety_score = self.safety_scorer.calculate_score(
            tracked_objects=tracked_objects,
            lane_result=lane_result,
            scene_analysis=None,  # Would pass SceneAnalysis if available
            scene_context=scene_context,
            warnings=[],  # Would pass active warnings
            ego_speed=None  # Would pass ego speed if available
        )

        logger.info(
            f"Safety Score: {safety_score.overall_score:.1f}/100 "
            f"({safety_score.safety_level.value})"
        )

        # Check for low safety score
        if safety_score.overall_score < 60:
            logger.warning(
                f"⚠ Low safety score: {safety_score.overall_score:.1f}"
            )

        # 8. Update trip analytics
        detection_counts = {}
        for obj in tracked_objects:
            class_name = obj.class_name if hasattr(obj, 'class_name') else 'unknown'
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        self.trip_analyzer.update_trip(
            detections=detection_counts,
            tracked_object_ids=[obj.track_id for obj in tracked_objects],
            scene_type=scene_context.road_type.value,
            weather=scene_context.weather.value,
            safety_score=safety_score.overall_score,
            fps=30.0,  # Would get actual FPS
            processing_time_ms=0.0,  # Would get actual processing time
            near_misses=safety_score.near_misses_count
        )

        # 9. Trigger recording events based on safety
        if safety_score.near_misses_count > 0:
            event = RecordingEvent(
                timestamp=timestamp,
                event_type="near_miss",
                severity="high",
                description=f"Near miss detected (safety score: {safety_score.overall_score:.1f})"
            )
            self.recorder.trigger_event(event)

        # Return enhanced results
        return {
            'traffic_lights': traffic_lights,
            'depth_map': depth_map,
            'segmentation_mask': segmentation_mask,
            'drivable_area': drivable_area,
            'scene_context': scene_context,
            'safety_score': safety_score,
            'class_percentages': class_percentages
        }

    def start_trip(self, trip_name: str = None):
        """Start a new trip with recording and analytics."""
        logger.info("=" * 80)
        logger.info("STARTING NEW TRIP")
        logger.info("=" * 80)

        # Start recording session
        session_id = self.recorder.start_session_recording(trip_name)
        logger.info(f"📹 Recording session started: {session_id}")

        # Start trip analytics
        trip_id = self.trip_analyzer.start_trip(trip_name)
        logger.info(f"📊 Trip analytics started: {trip_id}")

        # Reset safety scorer
        self.safety_scorer.reset_session()
        logger.info("🛡️ Safety scorer reset")

        return trip_id

    def end_trip(self):
        """End trip and generate reports."""
        logger.info("=" * 80)
        logger.info("ENDING TRIP")
        logger.info("=" * 80)

        # Stop recording
        session_info = self.recorder.stop_session_recording()
        if session_info:
            logger.info(
                f"📹 Recording saved: {session_info['duration_sec']:.1f}s, "
                f"{len(session_info['recorded_files'])} cameras"
            )

        # End trip analytics
        trip_stats = self.trip_analyzer.end_trip()
        if trip_stats:
            logger.info("📊 Trip Statistics:")
            logger.info(f"   Duration: {trip_stats.duration_seconds/60:.1f} minutes")
            logger.info(f"   Detections: {trip_stats.total_detections}")
            logger.info(f"   Unique objects: {trip_stats.unique_objects_tracked}")
            logger.info(f"   Avg safety score: {trip_stats.average_safety_score:.1f}")
            logger.info(f"   Near misses: {trip_stats.near_miss_count}")

        # Print safety statistics
        safety_stats = self.safety_scorer.get_statistics()
        logger.info("🛡️ Safety Statistics:")
        logger.info(f"   Average score: {safety_stats['average_score']:.1f}")
        logger.info(f"   Trend: {safety_stats['trend']}")
        logger.info(f"   Near misses: {safety_stats['total_near_misses']}")
        logger.info(f"   Lane departures: {safety_stats['total_lane_departures']}")

        # Print re-ID statistics
        reid_stats = self.reidentifier.get_statistics()
        logger.info("🔍 Re-Identification Statistics:")
        logger.info(f"   Total re-IDs: {reid_stats['total_reidentifications']}")
        logger.info(f"   Objects tracked: {reid_stats['objects_in_database']}")

        logger.info("=" * 80)

    def create_visualization(
        self,
        frame: np.ndarray,
        results: dict
    ) -> np.ndarray:
        """
        Create enhanced visualization with all features.

        Args:
            frame: Original frame
            results: Results from process_frame()

        Returns:
            Annotated frame
        """
        vis = frame.copy()

        # 1. Draw traffic lights
        if results.get('traffic_lights'):
            for light in results['traffic_lights']:
                x1, y1, x2, y2 = light.bbox
                color = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0)
                }.get(light.state.value, (255, 255, 255))

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis,
                    f"Light: {light.state.value}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # 2. Show scene info
        scene_context = results.get('scene_context')
        if scene_context:
            info_text = [
                f"Weather: {scene_context.weather.value}",
                f"Road: {scene_context.road_type.value}",
                f"Time: {scene_context.time_of_day.value}",
                f"Visibility: {scene_context.visibility_score:.2f}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(
                    vis,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                y_offset += 20

        # 3. Show safety score
        safety_score = results.get('safety_score')
        if safety_score:
            score_text = f"Safety: {safety_score.overall_score:.0f}/100"
            score_color = {
                'excellent': (0, 255, 0),
                'good': (0, 255, 255),
                'fair': (0, 165, 255),
                'poor': (0, 100, 255),
                'critical': (0, 0, 255)
            }.get(safety_score.safety_level.value, (255, 255, 255))

            cv2.putText(
                vis,
                score_text,
                (vis.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                score_color,
                2
            )

        # 4. Overlay depth map (side-by-side or blended)
        if results.get('depth_map') is not None:
            depth_colored = self.depth_estimator.create_colored_depth_map(
                results['depth_map']
            )
            # Blend with original (optional)
            # vis = cv2.addWeighted(vis, 0.7, depth_colored, 0.3, 0)

        return vis


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("ADVANCED ADAS FEATURES DEMONSTRATION")
    print("Version 1.1.0")
    print("=" * 80 + "\n")

    # Initialize demo
    demo = AdvancedADASDemo()

    print("\n✓ All systems initialized")
    print("\nThis demo shows how to integrate:")
    print("  📹 Multi-camera recording with event triggers")
    print("  🚦 Traffic light detection")
    print("  🌊 Monocular depth estimation")
    print("  🎨 Semantic segmentation")
    print("  🌤️ Scene recognition (weather, lighting, road type)")
    print("  🔍 Object re-identification across cameras")
    print("  🛡️ Real-time safety scoring")
    print("  📊 Trip analytics and reporting")

    print("\n" + "-" * 80)
    print("USAGE EXAMPLE:")
    print("-" * 80)

    # Example: Start a trip
    print("\n# Start a trip")
    demo.start_trip("demo_trip")

    # Example: Process frames (simulated)
    print("\n# Processing frames...")
    print("(In real usage, you would process frames from your cameras)")

    # Simulate a few frames
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_tracked_objects = []

    for i in range(3):
        print(f"\nFrame {i+1}:")
        results = demo.process_frame(
            camera_id=0,
            frame=dummy_frame,
            tracked_objects=dummy_tracked_objects
        )

        time.sleep(0.1)  # Simulate frame rate

    # End trip
    print("\n# End trip and generate reports")
    demo.end_trip()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

    print("\nGenerated outputs:")
    print("  • Video recordings: data/recordings/")
    print("  • Trip reports: data/analytics/")
    print("  • Event clips: data/recordings/events/")

    print("\nIntegration tips:")
    print("  1. Call demo.start_trip() at the beginning of each drive")
    print("  2. Call demo.process_frame() for each camera frame")
    print("  3. Call demo.end_trip() when finished to save reports")
    print("  4. Use demo.create_visualization() to see all features overlaid")
    print("\nSee examples/advanced_features_demo.py for full code\n")


if __name__ == "__main__":
    main()
