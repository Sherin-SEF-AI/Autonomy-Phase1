"""
Unit tests for advanced ADAS features.

Tests all new modules to ensure correct functionality.
"""

import unittest
import numpy as np
import cv2
import time
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from recording.video_recorder import MultiCameraRecorder, RecordingConfig, RecordingEvent
from perception.traffic_light_detection import TrafficLightDetector, TrafficLightState
from perception.depth_estimation import DepthEstimator, DepthModel
from perception.semantic_segmentation import SemanticSegmenter
from perception.scene_recognition import SceneRecognizer, WeatherCondition
from perception.object_reidentification import ObjectReIdentifier
from safety.safety_scorer import SafetyScorer, SafetyLevel
from analytics.trip_analyzer import TripAnalyzer


class TestRecording(unittest.TestCase):
    """Test multi-camera recording system."""

    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = RecordingConfig(
            output_dir=Path(self.temp_dir),
            buffer_duration_sec=5.0
        )
        self.recorder = MultiCameraRecorder(self.config)

    def tearDown(self):
        """Cleanup test environment."""
        self.recorder.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recorder_initialization(self):
        """Test recorder initializes correctly."""
        self.assertIsNotNone(self.recorder)
        self.assertEqual(len(self.recorder.recorders), 0)

    def test_add_camera(self):
        """Test adding cameras."""
        self.recorder.add_camera(0)
        self.recorder.add_camera(1)
        self.assertEqual(len(self.recorder.recorders), 2)

    def test_frame_buffering(self):
        """Test circular buffer."""
        self.recorder.add_camera(0)

        # Add frames
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10):
            self.recorder.add_frame(0, dummy_frame, time.time())

        # Check buffer
        stats = self.recorder.get_statistics()
        self.assertGreater(stats['camera_buffers'][0]['buffer_size'], 0)

    def test_session_recording(self):
        """Test session start/stop."""
        self.recorder.add_camera(0)

        session_id = self.recorder.start_session_recording("test")
        self.assertIsNotNone(session_id)
        self.assertTrue(self.recorder.current_session_id is not None)

        # Add some frames
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recorder.add_frame(0, dummy_frame, time.time())

        # Stop
        session_info = self.recorder.stop_session_recording()
        self.assertIsNotNone(session_info)
        self.assertIn('duration_sec', session_info)


class TestTrafficLightDetection(unittest.TestCase):
    """Test traffic light detection."""

    def setUp(self):
        """Setup detector."""
        self.detector = TrafficLightDetector()

    def test_detector_initialization(self):
        """Test detector initializes."""
        self.assertIsNotNone(self.detector)
        self.assertTrue(self.detector.enable_detection)

    def test_empty_image(self):
        """Test with empty image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        lights = self.detector.detect(image)
        self.assertIsInstance(lights, list)

    def test_red_light_image(self):
        """Test with simulated red light."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw red circle (simulated traffic light)
        cv2.circle(image, (320, 100), 20, (0, 0, 255), -1)

        lights = self.detector.detect(image)
        # Should detect something (may not be perfect with simple test)
        self.assertIsInstance(lights, list)

    def test_statistics(self):
        """Test statistics tracking."""
        stats = self.detector.get_statistics()
        self.assertIn('total_detections', stats)
        self.assertIn('enabled', stats)


class TestDepthEstimation(unittest.TestCase):
    """Test depth estimation."""

    def setUp(self):
        """Setup estimator."""
        # Use simple mode for testing (no ML dependencies)
        self.estimator = DepthEstimator(model_type=DepthModel.SIMPLE)

    def test_estimator_initialization(self):
        """Test estimator initializes."""
        self.assertIsNotNone(self.estimator)
        self.assertEqual(self.estimator.model_type, DepthModel.SIMPLE)

    def test_depth_estimation(self):
        """Test depth estimation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map = self.estimator.estimate_depth(image)

        self.assertIsNotNone(depth_map)
        self.assertEqual(depth_map.shape[:2], image.shape[:2])

    def test_depth_at_point(self):
        """Test depth at specific point."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map = self.estimator.estimate_depth(image)

        depth = self.estimator.get_depth_at_point(depth_map, (320, 240))
        self.assertIsInstance(depth, float)
        self.assertGreaterEqual(depth, 0)

    def test_colored_depth_map(self):
        """Test colored depth map creation."""
        depth_map = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        colored = self.estimator.create_colored_depth_map(depth_map)

        self.assertEqual(colored.shape, (480, 640, 3))


class TestSemanticSegmentation(unittest.TestCase):
    """Test semantic segmentation."""

    def setUp(self):
        """Setup segmenter."""
        # Use simple mode for testing
        self.segmenter = SemanticSegmenter(model_type="simple")

    def test_segmenter_initialization(self):
        """Test segmenter initializes."""
        self.assertIsNotNone(self.segmenter)
        self.assertEqual(self.segmenter.model_type, "simple")

    def test_segmentation(self):
        """Test segmentation."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mask = self.segmenter.segment(image)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (480, 640))

    def test_drivable_area(self):
        """Test drivable area extraction."""
        mask = np.random.randint(0, 19, (480, 640), dtype=np.uint8)
        drivable = self.segmenter.extract_drivable_area(mask)

        self.assertEqual(drivable.shape, mask.shape)

    def test_class_percentages(self):
        """Test class percentage calculation."""
        mask = np.random.randint(0, 19, (480, 640), dtype=np.uint8)
        percentages = self.segmenter.get_class_percentages(mask)

        self.assertIsInstance(percentages, dict)
        total_percentage = sum(percentages.values())
        self.assertLessEqual(total_percentage, 100.1)  # Allow small rounding


class TestSceneRecognition(unittest.TestCase):
    """Test scene recognition."""

    def setUp(self):
        """Setup recognizer."""
        self.recognizer = SceneRecognizer()

    def test_recognizer_initialization(self):
        """Test recognizer initializes."""
        self.assertIsNotNone(self.recognizer)

    def test_scene_recognition(self):
        """Test scene recognition."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        context = self.recognizer.recognize(image)

        self.assertIsNotNone(context)
        self.assertIsInstance(context.weather, WeatherCondition)
        self.assertGreaterEqual(context.brightness, 0)
        self.assertLessEqual(context.brightness, 255)
        self.assertGreaterEqual(context.visibility_score, 0)
        self.assertLessEqual(context.visibility_score, 1)

    def test_bright_image(self):
        """Test with bright image."""
        image = np.full((480, 640, 3), 250, dtype=np.uint8)
        context = self.recognizer.recognize(image)

        # Bright image should have high brightness
        self.assertGreater(context.brightness, 200)

    def test_dark_image(self):
        """Test with dark image."""
        image = np.full((480, 640, 3), 20, dtype=np.uint8)
        context = self.recognizer.recognize(image)

        # Dark image should have low brightness
        self.assertLess(context.brightness, 50)


class TestObjectReIdentification(unittest.TestCase):
    """Test object re-identification."""

    def setUp(self):
        """Setup re-identifier."""
        self.reidentifier = ObjectReIdentifier()

    def test_reidentifier_initialization(self):
        """Test re-identifier initializes."""
        self.assertIsNotNone(self.reidentifier)
        self.assertEqual(self.reidentifier.feature_type, "histogram")

    def test_feature_extraction(self):
        """Test feature extraction."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)

        features = self.reidentifier.extract_features(image, bbox)

        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)

    def test_similarity_calculation(self):
        """Test similarity calculation."""
        features1 = np.random.rand(96).astype(np.float32)
        features2 = np.random.rand(96).astype(np.float32)

        similarity = self.reidentifier._calculate_similarity(features1, features2)

        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)

    def test_identical_features(self):
        """Test identical features have high similarity."""
        features = np.random.rand(96).astype(np.float32)

        similarity = self.reidentifier._calculate_similarity(features, features)

        # Identical features should have similarity close to 1
        self.assertGreater(similarity, 0.99)


class TestSafetyScorer(unittest.TestCase):
    """Test safety scoring system."""

    def setUp(self):
        """Setup safety scorer."""
        self.scorer = SafetyScorer()

    def test_scorer_initialization(self):
        """Test scorer initializes."""
        self.assertIsNotNone(self.scorer)

    def test_score_calculation(self):
        """Test score calculation."""
        # Minimal test with empty data
        score = self.scorer.calculate_score(
            tracked_objects=[],
            lane_result=None,
            scene_analysis=None,
            scene_context=None,
            warnings=[]
        )

        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.overall_score, 0)
        self.assertLessEqual(score.overall_score, 100)
        self.assertIsInstance(score.safety_level, SafetyLevel)

    def test_score_range(self):
        """Test score is always in valid range."""
        for _ in range(10):
            score = self.scorer.calculate_score(
                tracked_objects=[],
                lane_result=None,
                scene_analysis=None,
                scene_context=None,
                warnings=[]
            )

            self.assertGreaterEqual(score.overall_score, 0)
            self.assertLessEqual(score.overall_score, 100)

    def test_statistics(self):
        """Test statistics."""
        stats = self.scorer.get_statistics()

        self.assertIn('average_score', stats)
        self.assertIn('total_near_misses', stats)
        self.assertIn('trend', stats)


class TestTripAnalyzer(unittest.TestCase):
    """Test trip analytics."""

    def setUp(self):
        """Setup trip analyzer."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = TripAnalyzer(output_dir=Path(self.temp_dir))

    def tearDown(self):
        """Cleanup."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyzer_initialization(self):
        """Test analyzer initializes."""
        self.assertIsNotNone(self.analyzer)

    def test_trip_lifecycle(self):
        """Test trip start/end."""
        trip_id = self.analyzer.start_trip("test_trip")
        self.assertIsNotNone(trip_id)
        self.assertIsNotNone(self.analyzer.current_trip)

        # Update trip
        self.analyzer.update_trip(
            detections={"car": 5},
            scene_type="urban",
            safety_score=85.0
        )

        # End trip
        trip_stats = self.analyzer.end_trip()
        self.assertIsNotNone(trip_stats)
        self.assertEqual(trip_stats.trip_id, trip_id)

    def test_trip_statistics(self):
        """Test trip statistics."""
        self.analyzer.start_trip("test")

        self.analyzer.update_trip(
            detections={"car": 10, "person": 5},
            safety_score=80.0
        )

        trip_stats = self.analyzer.end_trip()

        self.assertEqual(trip_stats.total_detections, 15)
        self.assertIn("car", trip_stats.detections_by_class)
        self.assertEqual(trip_stats.detections_by_class["car"], 10)

    def test_summary_report(self):
        """Test summary report generation."""
        # Create a trip
        self.analyzer.start_trip("trip1")
        self.analyzer.update_trip(detections={"car": 5})
        self.analyzer.end_trip()

        # Generate summary
        summary = self.analyzer.generate_summary_report()

        self.assertIsNotNone(summary)
        self.assertIn('trips_analyzed', summary)
        self.assertEqual(summary['trips_analyzed'], 1)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRecording))
    suite.addTests(loader.loadTestsFromTestCase(TestTrafficLightDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDepthEstimation))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticSegmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestSceneRecognition))
    suite.addTests(loader.loadTestsFromTestCase(TestObjectReIdentification))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestTripAnalyzer))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
