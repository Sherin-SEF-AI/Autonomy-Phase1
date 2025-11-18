"""
Unit tests for safety warning systems.

Tests FCW, LDW, BSW, and SafetyMonitor modules.
"""

import pytest
from safety import (
    ForwardCollisionWarning, FCWConfig, CollisionRisk,
    LaneDepartureWarning, LDWConfig, DepartureRisk, DepartureSide,
    BlindSpotWarning, BSWConfig, BlindSpotRisk, BlindSpotSide,
    SafetyMonitor, SafetyStatus
)
from utils.data_structures import TrackedObject, LaneInfo


class TestForwardCollisionWarning:
    """Tests for Forward Collision Warning system."""

    def test_fcw_initialization(self):
        """Test FCW initialization."""
        fcw = ForwardCollisionWarning()
        assert fcw.config is not None
        assert fcw.ego_velocity == 0.0
        assert fcw.warnings_issued == 0

    def test_fcw_no_risk_when_no_objects(self):
        """Test FCW returns no risk when no objects."""
        fcw = ForwardCollisionWarning()
        fcw.set_ego_velocity(10.0)  # 10 m/s

        risk, objects = fcw.analyze([])

        assert risk == CollisionRisk.NONE
        assert len(objects) == 0

    def test_fcw_detects_object_ahead(self):
        """Test FCW detects object directly ahead."""
        fcw = ForwardCollisionWarning()
        fcw.set_ego_velocity(15.0)  # 15 m/s = 54 km/h

        # Create object 20m ahead, stationary
        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(20.0, 0.0, 0.0),  # 20m ahead, centered
            current_velocity=(0.0, 0.0),  # Stationary
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=100.0
        )

        risk, objects = fcw.analyze([obj])

        # TTC = 20m / 15m/s = 1.33s -> HIGH risk
        assert risk in [CollisionRisk.HIGH, CollisionRisk.CRITICAL]
        assert len(objects) > 0

    def test_fcw_ignores_objects_behind(self):
        """Test FCW ignores objects behind vehicle."""
        fcw = ForwardCollisionWarning()
        fcw.set_ego_velocity(10.0)

        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(-5.0, 0.0, 0.0),  # 5m behind
            current_velocity=(0.0, 0.0),
            confidence=0.9,
            age=5,
            times_seen=5,
            last_seen=100.0
        )

        risk, objects = fcw.analyze([obj])

        assert risk == CollisionRisk.NONE
        assert len(objects) == 0

    def test_fcw_statistics(self):
        """Test FCW statistics tracking."""
        fcw = ForwardCollisionWarning()
        fcw.set_ego_velocity(15.0)

        stats = fcw.get_statistics()

        assert 'warnings_issued' in stats
        assert 'ego_velocity' in stats
        assert stats['ego_velocity'] == 15.0


class TestLaneDepartureWarning:
    """Tests for Lane Departure Warning system."""

    def test_ldw_initialization(self):
        """Test LDW initialization."""
        ldw = LaneDepartureWarning()
        assert ldw.config is not None
        assert ldw.ego_velocity == 0.0
        assert ldw.warnings_issued == 0

    def test_ldw_no_risk_centered_in_lane(self):
        """Test LDW shows no risk when centered."""
        ldw = LaneDepartureWarning()
        ldw.set_ego_velocity(15.0)

        lane_info = LaneInfo(
            left_lane_detected=True,
            right_lane_detected=True,
            left_coeffs=[0.0, -1.75, 200],
            right_coeffs=[0.0, 1.75, 200],
            lateral_offset=0.0,  # Centered
            lane_width=3.5,
            departure_warning="",
            confidence=0.9
        )

        risk, side = ldw.analyze(lane_info)

        assert risk == DepartureRisk.NONE
        assert side == DepartureSide.NONE

    def test_ldw_detects_left_departure(self):
        """Test LDW detects left lane departure."""
        ldw = LaneDepartureWarning()
        ldw.set_ego_velocity(15.0)

        lane_info = LaneInfo(
            left_lane_detected=True,
            right_lane_detected=True,
            left_coeffs=[0.0, -1.75, 200],
            right_coeffs=[0.0, 1.75, 200],
            lateral_offset=0.8,  # 80cm to the left
            lane_width=3.5,
            departure_warning="",
            confidence=0.9
        )

        risk, side = ldw.analyze(lane_info)

        assert risk in [DepartureRisk.HIGH, DepartureRisk.CRITICAL]
        assert side == DepartureSide.LEFT

    def test_ldw_detects_right_departure(self):
        """Test LDW detects right lane departure."""
        ldw = LaneDepartureWarning()
        ldw.set_ego_velocity(15.0)

        lane_info = LaneInfo(
            left_lane_detected=True,
            right_lane_detected=True,
            left_coeffs=[0.0, -1.75, 200],
            right_coeffs=[0.0, 1.75, 200],
            lateral_offset=-0.8,  # 80cm to the right
            lane_width=3.5,
            departure_warning="",
            confidence=0.9
        )

        risk, side = ldw.analyze(lane_info)

        assert risk in [DepartureRisk.HIGH, DepartureRisk.CRITICAL]
        assert side == DepartureSide.RIGHT

    def test_ldw_no_warning_low_speed(self):
        """Test LDW doesn't warn at low speeds."""
        ldw = LaneDepartureWarning()
        ldw.set_ego_velocity(5.0)  # Below threshold

        lane_info = LaneInfo(
            left_lane_detected=True,
            right_lane_detected=True,
            left_coeffs=[0.0, -1.75, 200],
            right_coeffs=[0.0, 1.75, 200],
            lateral_offset=0.8,  # Significant offset
            lane_width=3.5,
            departure_warning="",
            confidence=0.9
        )

        risk, side = ldw.analyze(lane_info)

        assert risk == DepartureRisk.NONE


class TestBlindSpotWarning:
    """Tests for Blind Spot Warning system."""

    def test_bsw_initialization(self):
        """Test BSW initialization."""
        bsw = BlindSpotWarning()
        assert bsw.config is not None
        assert bsw.ego_velocity == 0.0
        assert bsw.warnings_issued == 0

    def test_bsw_no_risk_no_objects(self):
        """Test BSW shows no risk when no objects."""
        bsw = BlindSpotWarning()
        bsw.set_ego_velocity(15.0)

        risk, side, objects = bsw.analyze([])

        assert risk == BlindSpotRisk.NONE
        assert side == BlindSpotSide.NONE
        assert len(objects) == 0

    def test_bsw_detects_left_blind_spot(self):
        """Test BSW detects object in left blind spot."""
        bsw = BlindSpotWarning()
        bsw.set_ego_velocity(15.0)

        # Object in left blind spot (beside vehicle)
        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(0.0, 2.5, 0.0),  # Directly to the left
            current_velocity=(0.0, 0.0),
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=100.0
        )

        risk, side, objects = bsw.analyze([obj])

        assert risk in [BlindSpotRisk.HIGH, BlindSpotRisk.CRITICAL]
        assert side in [BlindSpotSide.LEFT, BlindSpotSide.BOTH]
        assert len(objects) > 0

    def test_bsw_detects_right_blind_spot(self):
        """Test BSW detects object in right blind spot."""
        bsw = BlindSpotWarning()
        bsw.set_ego_velocity(15.0)

        # Object in right blind spot
        obj = TrackedObject(
            track_id=1,
            class_name="motorcycle",
            current_position=(0.0, -2.5, 0.0),  # Directly to the right
            current_velocity=(0.0, 0.0),
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=100.0
        )

        risk, side, objects = bsw.analyze([obj])

        assert risk in [BlindSpotRisk.HIGH, BlindSpotRisk.CRITICAL]
        assert side in [BlindSpotSide.RIGHT, BlindSpotSide.BOTH]
        assert len(objects) > 0


class TestSafetyMonitor:
    """Tests for unified SafetyMonitor."""

    def test_safety_monitor_initialization(self):
        """Test SafetyMonitor initialization."""
        monitor = SafetyMonitor()
        assert monitor.fcw is not None
        assert monitor.ldw is not None
        assert monitor.bsw is not None
        assert monitor.current_status == SafetyStatus.SAFE

    def test_safety_monitor_safe_status(self):
        """Test SafetyMonitor shows safe with no threats."""
        monitor = SafetyMonitor()
        monitor.set_ego_velocity(15.0)

        status = monitor.update(tracked_objects=[], lane_info=None)

        assert status == SafetyStatus.SAFE
        assert len(monitor.get_active_alerts()) == 0

    def test_safety_monitor_collision_warning(self):
        """Test SafetyMonitor detects collision risk."""
        monitor = SafetyMonitor()
        monitor.set_ego_velocity(20.0)

        # Object 15m ahead, stationary (TTC = 0.75s -> CRITICAL)
        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(15.0, 0.0, 0.0),
            current_velocity=(0.0, 0.0),
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=100.0
        )

        status = monitor.update(tracked_objects=[obj], lane_info=None)

        assert status in [SafetyStatus.WARNING, SafetyStatus.CRITICAL]
        assert len(monitor.get_active_alerts()) > 0

        # Check alert type
        alerts = monitor.get_active_alerts()
        assert any(alert.alert_type == "FCW" for alert in alerts)

    def test_safety_monitor_statistics(self):
        """Test SafetyMonitor statistics."""
        monitor = SafetyMonitor()

        stats = monitor.get_statistics()

        assert 'current_status' in stats
        assert 'active_alerts' in stats
        assert 'total_alerts' in stats
        assert 'alerts_by_type' in stats
        assert 'fcw' in stats
        assert 'ldw' in stats
        assert 'bsw' in stats

    def test_safety_monitor_reset(self):
        """Test SafetyMonitor reset."""
        monitor = SafetyMonitor()
        monitor.set_ego_velocity(15.0)

        # Generate some alerts
        obj = TrackedObject(
            track_id=1,
            class_name="car",
            current_position=(10.0, 0.0, 0.0),
            current_velocity=(0.0, 0.0),
            confidence=0.9,
            age=10,
            times_seen=10,
            last_seen=100.0
        )
        monitor.update(tracked_objects=[obj], lane_info=None)

        # Reset
        monitor.reset()

        # Check all cleared
        assert monitor.current_status == SafetyStatus.SAFE
        assert len(monitor.get_active_alerts()) == 0
        assert monitor.total_alerts == 0


# Run tests with: pytest tests/test_safety_systems.py -v
