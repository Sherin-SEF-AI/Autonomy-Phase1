"""
Safety Monitor - Coordinates all safety warning systems.

Integrates FCW, LDW, and BSW systems and provides unified safety monitoring
and alert management.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

from utils.data_structures import TrackedObject, LaneInfo
from utils.logger import get_logger
from .collision_warning import ForwardCollisionWarning, FCWConfig, CollisionRisk
from .lane_departure_warning import LaneDepartureWarning, LDWConfig, DepartureRisk, DepartureSide
from .blind_spot_warning import BlindSpotWarning, BSWConfig, BlindSpotRisk, BlindSpotSide


logger = get_logger()


class SafetyStatus(Enum):
    """Overall safety status."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SafetyAlert:
    """Safety alert information."""
    timestamp: float
    alert_type: str  # "FCW", "LDW", "BSW"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    details: Dict[str, Any]


class SafetyMonitor:
    """
    Unified safety monitoring system.

    Coordinates FCW, LDW, and BSW systems to provide comprehensive
    safety monitoring and alerts.
    """

    def __init__(
        self,
        fcw_config: Optional[FCWConfig] = None,
        ldw_config: Optional[LDWConfig] = None,
        bsw_config: Optional[BSWConfig] = None,
        enable_fcw: bool = True,
        enable_ldw: bool = True,
        enable_bsw: bool = True
    ):
        """
        Initialize safety monitor.

        Args:
            fcw_config: FCW configuration
            ldw_config: LDW configuration
            bsw_config: BSW configuration
            enable_fcw: Enable Forward Collision Warning
            enable_ldw: Enable Lane Departure Warning
            enable_bsw: Enable Blind Spot Warning
        """
        # Initialize warning systems
        self.fcw = ForwardCollisionWarning(fcw_config) if enable_fcw else None
        self.ldw = LaneDepartureWarning(ldw_config) if enable_ldw else None
        self.bsw = BlindSpotWarning(bsw_config) if enable_bsw else None

        # State
        self.ego_velocity = 0.0  # m/s
        self.lateral_velocity = 0.0  # m/s

        # Alert management
        self.active_alerts: List[SafetyAlert] = []
        self.alert_history: List[SafetyAlert] = []
        self.max_history = 100

        # Overall status
        self.current_status = SafetyStatus.SAFE

        # Statistics
        self.total_alerts = 0
        self.alerts_by_type = {
            'FCW': 0,
            'LDW': 0,
            'BSW': 0
        }

        logger.info(
            f"Safety Monitor initialized (FCW: {enable_fcw}, LDW: {enable_ldw}, BSW: {enable_bsw})"
        )

    def set_ego_velocity(self, velocity: float, lateral_velocity: float = 0.0):
        """
        Set ego vehicle velocity.

        Args:
            velocity: Longitudinal velocity in m/s
            lateral_velocity: Lateral velocity in m/s
        """
        self.ego_velocity = velocity
        self.lateral_velocity = lateral_velocity

        # Update all systems
        if self.fcw:
            self.fcw.set_ego_velocity(velocity)
        if self.ldw:
            self.ldw.set_ego_velocity(velocity, lateral_velocity)
        if self.bsw:
            self.bsw.set_ego_velocity(velocity)

    def update(
        self,
        tracked_objects: Optional[List[TrackedObject]] = None,
        lane_info: Optional[LaneInfo] = None
    ) -> SafetyStatus:
        """
        Update all safety systems and get overall status.

        Args:
            tracked_objects: List of tracked objects
            lane_info: Lane detection information

        Returns:
            Overall safety status
        """
        # Clear previous alerts
        self.active_alerts.clear()

        # Run FCW analysis
        if self.fcw and tracked_objects:
            fcw_risk, fcw_objects = self.fcw.analyze(tracked_objects, self.ego_velocity)
            if fcw_risk != CollisionRisk.NONE:
                alert = SafetyAlert(
                    timestamp=time.time(),
                    alert_type="FCW",
                    severity=self._map_collision_risk(fcw_risk),
                    message=f"Forward collision risk: {fcw_risk.name}",
                    details={
                        'risk_level': fcw_risk.name,
                        'num_objects': len(fcw_objects),
                        'objects': [obj.track_id for obj in fcw_objects]
                    }
                )
                self._add_alert(alert)

        # Run LDW analysis
        if self.ldw and lane_info:
            ldw_risk, departure_side = self.ldw.analyze(
                lane_info,
                self.ego_velocity,
                self.lateral_velocity
            )
            if ldw_risk != DepartureRisk.NONE:
                alert = SafetyAlert(
                    timestamp=time.time(),
                    alert_type="LDW",
                    severity=self._map_departure_risk(ldw_risk),
                    message=f"Lane departure: {departure_side.value} - {ldw_risk.name}",
                    details={
                        'risk_level': ldw_risk.name,
                        'departure_side': departure_side.value,
                        'lateral_offset': lane_info.lateral_offset
                    }
                )
                self._add_alert(alert)

        # Run BSW analysis
        if self.bsw and tracked_objects:
            bsw_risk, bsw_side, bsw_objects = self.bsw.analyze(
                tracked_objects,
                self.ego_velocity
            )
            if bsw_risk != BlindSpotRisk.NONE:
                alert = SafetyAlert(
                    timestamp=time.time(),
                    alert_type="BSW",
                    severity=self._map_blindspot_risk(bsw_risk),
                    message=f"Blind spot warning: {bsw_side.value} - {bsw_risk.name}",
                    details={
                        'risk_level': bsw_risk.name,
                        'affected_side': bsw_side.value,
                        'num_objects': len(bsw_objects),
                        'objects': [obj.track_id for obj in bsw_objects]
                    }
                )
                self._add_alert(alert)

        # Determine overall status
        self.current_status = self._determine_overall_status()

        return self.current_status

    def _add_alert(self, alert: SafetyAlert):
        """
        Add an alert to active alerts and history.

        Args:
            alert: Safety alert
        """
        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        # Maintain history size
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Update statistics
        self.total_alerts += 1
        self.alerts_by_type[alert.alert_type] += 1

    def _determine_overall_status(self) -> SafetyStatus:
        """
        Determine overall safety status based on active alerts.

        Returns:
            Overall safety status
        """
        if not self.active_alerts:
            return SafetyStatus.SAFE

        # Check for critical alerts
        for alert in self.active_alerts:
            if alert.severity == "critical":
                return SafetyStatus.CRITICAL

        # Check for high severity alerts
        high_severity = any(alert.severity == "high" for alert in self.active_alerts)
        if high_severity:
            return SafetyStatus.WARNING

        # Check for medium severity alerts
        medium_severity = any(alert.severity == "medium" for alert in self.active_alerts)
        if medium_severity:
            return SafetyStatus.CAUTION

        # Low severity alerts
        return SafetyStatus.CAUTION

    def _map_collision_risk(self, risk: CollisionRisk) -> str:
        """Map collision risk to severity level."""
        mapping = {
            CollisionRisk.NONE: "none",
            CollisionRisk.LOW: "low",
            CollisionRisk.MEDIUM: "medium",
            CollisionRisk.HIGH: "high",
            CollisionRisk.CRITICAL: "critical"
        }
        return mapping.get(risk, "none")

    def _map_departure_risk(self, risk: DepartureRisk) -> str:
        """Map departure risk to severity level."""
        mapping = {
            DepartureRisk.NONE: "none",
            DepartureRisk.LOW: "low",
            DepartureRisk.MEDIUM: "medium",
            DepartureRisk.HIGH: "high",
            DepartureRisk.CRITICAL: "critical"
        }
        return mapping.get(risk, "none")

    def _map_blindspot_risk(self, risk: BlindSpotRisk) -> str:
        """Map blind spot risk to severity level."""
        mapping = {
            BlindSpotRisk.NONE: "none",
            BlindSpotRisk.LOW: "low",
            BlindSpotRisk.MEDIUM: "medium",
            BlindSpotRisk.HIGH: "high",
            BlindSpotRisk.CRITICAL: "critical"
        }
        return mapping.get(risk, "none")

    def get_active_alerts(self) -> List[SafetyAlert]:
        """Get list of currently active alerts."""
        return self.active_alerts.copy()

    def get_alert_history(self, limit: Optional[int] = None) -> List[SafetyAlert]:
        """
        Get alert history.

        Args:
            limit: Optional limit on number of alerts

        Returns:
            List of recent alerts
        """
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get safety monitoring statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'current_status': self.current_status.value,
            'active_alerts': len(self.active_alerts),
            'total_alerts': self.total_alerts,
            'alerts_by_type': self.alerts_by_type.copy(),
            'ego_velocity': self.ego_velocity,
            'lateral_velocity': self.lateral_velocity
        }

        # Add per-system statistics
        if self.fcw:
            stats['fcw'] = self.fcw.get_statistics()
        if self.ldw:
            stats['ldw'] = self.ldw.get_statistics()
        if self.bsw:
            stats['bsw'] = self.bsw.get_statistics()

        return stats

    def reset(self):
        """Reset all safety systems."""
        if self.fcw:
            self.fcw.reset()
        if self.ldw:
            self.ldw.reset()
        if self.bsw:
            self.bsw.reset()

        self.active_alerts.clear()
        self.alert_history.clear()
        self.total_alerts = 0
        self.alerts_by_type = {
            'FCW': 0,
            'LDW': 0,
            'BSW': 0
        }
        self.current_status = SafetyStatus.SAFE

        logger.info("Safety Monitor reset")

    def enable_system(self, system: str, enabled: bool):
        """
        Enable or disable a specific safety system.

        Args:
            system: System name ("FCW", "LDW", "BSW")
            enabled: Enable or disable
        """
        if system == "FCW" and self.fcw:
            if not enabled:
                self.fcw = None
                logger.info("FCW disabled")
        elif system == "LDW" and self.ldw:
            if not enabled:
                self.ldw = None
                logger.info("LDW disabled")
        elif system == "BSW" and self.bsw:
            if not enabled:
                self.bsw = None
                logger.info("BSW disabled")

    def get_status(self) -> SafetyStatus:
        """Get current overall safety status."""
        return self.current_status
