"""
Scenario Testing Framework
Comprehensive testing framework for autonomous driving systems.

Features:
- Predefined test scenarios (highway, urban, parking, etc.)
- Custom scenario definition
- Traffic simulation
- Edge case and safety-critical testing
- Performance metrics and evaluation
- Pass/fail criteria
- Automated test execution
- Detailed test reports

Author: Autonomous Driving System
Version: 1.3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import time
import json
from collections import defaultdict
import math


class ScenarioType(Enum):
    """Types of test scenarios"""
    HIGHWAY_CRUISE = "highway_cruise"
    HIGHWAY_LANE_CHANGE = "highway_lane_change"
    URBAN_INTERSECTION = "urban_intersection"
    URBAN_PEDESTRIAN = "urban_pedestrian"
    PARKING = "parking"
    EMERGENCY_BRAKING = "emergency_braking"
    CUT_IN = "cut_in"
    MERGING = "merging"
    ROUNDABOUT = "roundabout"
    ADVERSE_WEATHER = "adverse_weather"
    NIGHT_DRIVING = "night_driving"
    CONSTRUCTION_ZONE = "construction_zone"
    CUSTOM = "custom"


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class MetricType(Enum):
    """Performance metric types"""
    SAFETY = "safety"
    COMFORT = "comfort"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    LATENCY = "latency"


class SeverityLevel(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Position2D:
    """2D position"""
    x: float
    y: float

    def distance_to(self, other: 'Position2D') -> float:
        """Calculate Euclidean distance"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Actor:
    """Scenario actor (vehicle, pedestrian, cyclist, etc.)"""
    actor_id: str
    actor_type: str  # vehicle, pedestrian, cyclist, etc.
    position: Position2D
    heading: float  # degrees
    speed: float  # m/s
    behavior: str = "stationary"  # stationary, constant_speed, accelerating, etc.


@dataclass
class ScenarioConfig:
    """Scenario configuration"""
    scenario_id: str
    scenario_type: ScenarioType
    description: str
    duration: float = 30.0  # seconds
    initial_ego_position: Position2D = field(default_factory=lambda: Position2D(0, 0))
    initial_ego_speed: float = 0.0  # m/s
    actors: List[Actor] = field(default_factory=list)
    weather_condition: str = "clear"  # clear, rain, fog, snow
    lighting_condition: str = "day"  # day, night, dusk, dawn
    road_condition: str = "dry"  # dry, wet, icy, snow
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCriteria:
    """Test pass/fail criteria"""
    name: str
    metric_type: MetricType
    threshold: float
    comparison: str = "less_than"  # less_than, greater_than, equal_to, within_range
    severity: SeverityLevel = SeverityLevel.HIGH
    description: str = ""

    def evaluate(self, value: float) -> bool:
        """Evaluate if criterion is met"""
        if self.comparison == "less_than":
            return value < self.threshold
        elif self.comparison == "greater_than":
            return value > self.threshold
        elif self.comparison == "equal_to":
            return abs(value - self.threshold) < 1e-6
        else:
            return False


@dataclass
class MetricResult:
    """Result of a performance metric"""
    name: str
    metric_type: MetricType
    value: float
    unit: str
    passed: bool
    threshold: Optional[float] = None


@dataclass
class TestIssue:
    """Test issue/violation"""
    timestamp: float
    severity: SeverityLevel
    category: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result of scenario execution"""
    scenario_id: str
    scenario_type: ScenarioType
    start_time: float
    end_time: float
    duration: float
    result: TestResult
    metrics: List[MetricResult] = field(default_factory=list)
    issues: List[TestIssue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_metric(self, metric: MetricResult):
        """Add a metric result"""
        self.metrics.append(metric)

    def add_issue(self, issue: TestIssue):
        """Add a test issue"""
        self.issues.append(issue)

        # Update overall result if critical/high severity
        if issue.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            if self.result == TestResult.PASSED:
                self.result = TestResult.FAILED

    def get_critical_issues(self) -> List[TestIssue]:
        """Get critical issues"""
        return [i for i in self.issues if i.severity == SeverityLevel.CRITICAL]

    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricResult]:
        """Get metrics of specific type"""
        return [m for m in self.metrics if m.metric_type == metric_type]


@dataclass
class TestSuiteResult:
    """Result of full test suite"""
    suite_name: str
    start_time: float
    end_time: float
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    warning_scenarios: int = 0
    skipped_scenarios: int = 0
    scenario_results: List[ScenarioResult] = field(default_factory=list)

    def add_scenario_result(self, result: ScenarioResult):
        """Add scenario result and update counts"""
        self.scenario_results.append(result)
        self.total_scenarios += 1

        if result.result == TestResult.PASSED:
            self.passed_scenarios += 1
        elif result.result == TestResult.FAILED:
            self.failed_scenarios += 1
        elif result.result == TestResult.WARNING:
            self.warning_scenarios += 1
        elif result.result == TestResult.SKIPPED:
            self.skipped_scenarios += 1

    def get_pass_rate(self) -> float:
        """Calculate pass rate"""
        if self.total_scenarios == 0:
            return 0.0
        return self.passed_scenarios / self.total_scenarios * 100.0


class ScenarioExecutor:
    """
    Scenario Test Executor

    Executes test scenarios and collects performance metrics.
    """

    def __init__(self):
        self.current_scenario: Optional[ScenarioConfig] = None
        self.current_result: Optional[ScenarioResult] = None
        self.elapsed_time = 0.0
        self.timestep = 0.1  # seconds

        # Metrics tracking
        self.collision_detected = False
        self.max_acceleration = 0.0
        self.max_jerk = 0.0
        self.max_lateral_acceleration = 0.0
        self.total_distance = 0.0
        self.speeds: List[float] = []
        self.accelerations: List[float] = []

        # Safety tracking
        self.min_distance_to_actor = float('inf')
        self.time_to_collision_violations = 0
        self.lane_departures = 0

        print("Scenario Executor initialized")

    def execute_scenario(self, scenario: ScenarioConfig,
                        criteria: List[TestCriteria]) -> ScenarioResult:
        """
        Execute a test scenario

        Args:
            scenario: Scenario configuration
            criteria: List of test criteria

        Returns:
            Scenario result
        """
        print(f"\n{'='*60}")
        print(f"Executing Scenario: {scenario.scenario_id}")
        print(f"Type: {scenario.scenario_type.value}")
        print(f"Description: {scenario.description}")
        print(f"{'='*60}\n")

        self.current_scenario = scenario
        self.current_result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_type=scenario.scenario_type,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            result=TestResult.PASSED
        )

        # Reset metrics
        self._reset_metrics()

        # Run scenario simulation
        self._simulate_scenario(scenario)

        # Evaluate criteria
        self._evaluate_criteria(criteria)

        # Finalize result
        self.current_result.end_time = time.time()
        self.current_result.duration = self.elapsed_time

        # Determine final result
        if self.collision_detected:
            self.current_result.add_issue(TestIssue(
                timestamp=self.elapsed_time,
                severity=SeverityLevel.CRITICAL,
                category="safety",
                description="Collision detected"
            ))

        print(f"\nScenario Result: {self.current_result.result.value.upper()}")
        print(f"Duration: {self.elapsed_time:.2f}s")
        print(f"Issues: {len(self.current_result.issues)}")

        return self.current_result

    def _reset_metrics(self):
        """Reset metrics for new scenario"""
        self.elapsed_time = 0.0
        self.collision_detected = False
        self.max_acceleration = 0.0
        self.max_jerk = 0.0
        self.max_lateral_acceleration = 0.0
        self.total_distance = 0.0
        self.speeds = []
        self.accelerations = []
        self.min_distance_to_actor = float('inf')
        self.time_to_collision_violations = 0
        self.lane_departures = 0

    def _simulate_scenario(self, scenario: ScenarioConfig):
        """
        Simulate scenario execution

        This is a simplified simulation. In production, this would interface
        with actual vehicle systems or a physics simulator.
        """
        # Initialize ego vehicle state
        ego_position = scenario.initial_ego_position
        ego_speed = scenario.initial_ego_speed
        ego_acceleration = 0.0
        previous_acceleration = 0.0

        num_steps = int(scenario.duration / self.timestep)

        for step in range(num_steps):
            current_time = step * self.timestep
            self.elapsed_time = current_time

            # Simulate scenario-specific behavior
            if scenario.scenario_type == ScenarioType.EMERGENCY_BRAKING:
                ego_acceleration = self._simulate_emergency_braking(current_time)
            elif scenario.scenario_type == ScenarioType.HIGHWAY_LANE_CHANGE:
                ego_acceleration = self._simulate_lane_change(current_time)
            elif scenario.scenario_type == ScenarioType.CUT_IN:
                ego_acceleration = self._simulate_cut_in(current_time, scenario.actors)
            else:
                # Default constant speed
                ego_acceleration = 0.0

            # Update vehicle state
            ego_speed += ego_acceleration * self.timestep
            ego_speed = max(0.0, ego_speed)  # No negative speed

            distance_traveled = ego_speed * self.timestep
            ego_position.x += distance_traveled

            self.total_distance += distance_traveled

            # Calculate jerk
            jerk = (ego_acceleration - previous_acceleration) / self.timestep

            # Track metrics
            self.speeds.append(ego_speed)
            self.accelerations.append(ego_acceleration)
            self.max_acceleration = max(self.max_acceleration, abs(ego_acceleration))
            self.max_jerk = max(self.max_jerk, abs(jerk))

            # Check distance to actors
            for actor in scenario.actors:
                distance = ego_position.distance_to(actor.position)
                self.min_distance_to_actor = min(self.min_distance_to_actor, distance)

                # Collision detection (simplified)
                if distance < 2.0:  # 2 meter threshold
                    self.collision_detected = True

                # Time-to-collision check
                if ego_speed > 0 and distance < 50.0:
                    ttc = distance / ego_speed
                    if ttc < 2.0:  # Less than 2 seconds
                        self.time_to_collision_violations += 1

            previous_acceleration = ego_acceleration

            # Progress indicator (every 5 seconds)
            if step % int(5.0 / self.timestep) == 0:
                print(f"  t={current_time:5.1f}s: speed={ego_speed:5.2f} m/s, "
                      f"accel={ego_acceleration:5.2f} m/s², pos={ego_position.x:6.1f}m")

    def _simulate_emergency_braking(self, t: float) -> float:
        """Simulate emergency braking scenario"""
        if t < 2.0:
            return 0.0  # Coast
        elif t < 4.0:
            return -6.0  # Emergency brake
        else:
            return 0.0  # Stopped

    def _simulate_lane_change(self, t: float) -> float:
        """Simulate lane change scenario"""
        # Gentle acceleration during lane change
        return 0.5 if 3.0 < t < 8.0 else 0.0

    def _simulate_cut_in(self, t: float, actors: List[Actor]) -> float:
        """Simulate cut-in scenario"""
        # Another vehicle cuts in at t=5s
        if 5.0 < t < 7.0:
            return -4.0  # Moderate braking
        else:
            return 0.0

    def _evaluate_criteria(self, criteria: List[TestCriteria]):
        """Evaluate test criteria"""
        for criterion in criteria:
            value = self._get_metric_value(criterion.name)

            passed = criterion.evaluate(value)

            # Add metric result
            metric_result = MetricResult(
                name=criterion.name,
                metric_type=criterion.metric_type,
                value=value,
                unit=self._get_metric_unit(criterion.name),
                passed=passed,
                threshold=criterion.threshold
            )

            self.current_result.add_metric(metric_result)

            # Add issue if failed
            if not passed:
                self.current_result.add_issue(TestIssue(
                    timestamp=self.elapsed_time,
                    severity=criterion.severity,
                    category=criterion.metric_type.value,
                    description=f"{criterion.name} failed: {value:.2f} (threshold: {criterion.threshold})",
                    details={'value': value, 'threshold': criterion.threshold}
                ))

    def _get_metric_value(self, metric_name: str) -> float:
        """Get value for a specific metric"""
        if metric_name == "max_acceleration":
            return self.max_acceleration
        elif metric_name == "max_jerk":
            return self.max_jerk
        elif metric_name == "max_lateral_acceleration":
            return self.max_lateral_acceleration
        elif metric_name == "min_distance_to_actor":
            return self.min_distance_to_actor
        elif metric_name == "average_speed":
            return np.mean(self.speeds) if self.speeds else 0.0
        elif metric_name == "total_distance":
            return self.total_distance
        elif metric_name == "ttc_violations":
            return float(self.time_to_collision_violations)
        elif metric_name == "lane_departures":
            return float(self.lane_departures)
        else:
            return 0.0

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for a specific metric"""
        units = {
            "max_acceleration": "m/s²",
            "max_jerk": "m/s³",
            "max_lateral_acceleration": "m/s²",
            "min_distance_to_actor": "m",
            "average_speed": "m/s",
            "total_distance": "m",
            "ttc_violations": "count",
            "lane_departures": "count"
        }
        return units.get(metric_name, "")


class ScenarioLibrary:
    """
    Scenario Library

    Predefined test scenarios for common driving situations.
    """

    @staticmethod
    def highway_cruise() -> Tuple[ScenarioConfig, List[TestCriteria]]:
        """Highway cruise control scenario"""
        scenario = ScenarioConfig(
            scenario_id="HWY_CRUISE_001",
            scenario_type=ScenarioType.HIGHWAY_CRUISE,
            description="Maintain steady speed on highway",
            duration=20.0,
            initial_ego_speed=25.0,  # 90 km/h
            actors=[
                Actor("lead_vehicle", "vehicle", Position2D(50, 0), 0, 25.0, "constant_speed")
            ]
        )

        criteria = [
            TestCriteria("max_acceleration", MetricType.COMFORT, 2.0, "less_than"),
            TestCriteria("max_jerk", MetricType.COMFORT, 3.0, "less_than"),
            TestCriteria("min_distance_to_actor", MetricType.SAFETY, 10.0, "greater_than",
                        severity=SeverityLevel.CRITICAL)
        ]

        return scenario, criteria

    @staticmethod
    def emergency_braking() -> Tuple[ScenarioConfig, List[TestCriteria]]:
        """Emergency braking scenario"""
        scenario = ScenarioConfig(
            scenario_id="EMERG_BRAKE_001",
            scenario_type=ScenarioType.EMERGENCY_BRAKING,
            description="Emergency braking to avoid collision",
            duration=10.0,
            initial_ego_speed=20.0,
            actors=[
                Actor("obstacle", "vehicle", Position2D(40, 0), 0, 0.0, "stationary")
            ]
        )

        criteria = [
            TestCriteria("min_distance_to_actor", MetricType.SAFETY, 5.0, "greater_than",
                        severity=SeverityLevel.CRITICAL),
            TestCriteria("max_acceleration", MetricType.SAFETY, 8.0, "less_than",
                        severity=SeverityLevel.HIGH)
        ]

        return scenario, criteria

    @staticmethod
    def lane_change() -> Tuple[ScenarioConfig, List[TestCriteria]]:
        """Highway lane change scenario"""
        scenario = ScenarioConfig(
            scenario_id="LANE_CHANGE_001",
            scenario_type=ScenarioType.HIGHWAY_LANE_CHANGE,
            description="Safe lane change on highway",
            duration=15.0,
            initial_ego_speed=22.0,
            actors=[
                Actor("adjacent_vehicle", "vehicle", Position2D(10, 3.5), 0, 22.0, "constant_speed")
            ]
        )

        criteria = [
            TestCriteria("max_lateral_acceleration", MetricType.COMFORT, 2.0, "less_than"),
            TestCriteria("min_distance_to_actor", MetricType.SAFETY, 5.0, "greater_than",
                        severity=SeverityLevel.HIGH),
            TestCriteria("lane_departures", MetricType.SAFETY, 0.0, "equal_to",
                        severity=SeverityLevel.MEDIUM)
        ]

        return scenario, criteria

    @staticmethod
    def cut_in() -> Tuple[ScenarioConfig, List[TestCriteria]]:
        """Vehicle cut-in scenario"""
        scenario = ScenarioConfig(
            scenario_id="CUT_IN_001",
            scenario_type=ScenarioType.CUT_IN,
            description="React to vehicle cutting in",
            duration=12.0,
            initial_ego_speed=20.0,
            actors=[
                Actor("cutting_vehicle", "vehicle", Position2D(25, 3.5), 0, 18.0, "cut_in")
            ]
        )

        criteria = [
            TestCriteria("min_distance_to_actor", MetricType.SAFETY, 8.0, "greater_than",
                        severity=SeverityLevel.CRITICAL),
            TestCriteria("max_acceleration", MetricType.COMFORT, 5.0, "less_than"),
            TestCriteria("ttc_violations", MetricType.SAFETY, 5.0, "less_than",
                        severity=SeverityLevel.HIGH)
        ]

        return scenario, criteria


class TestSuite:
    """
    Test Suite

    Manages and executes collections of test scenarios.
    """

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.scenarios: List[Tuple[ScenarioConfig, List[TestCriteria]]] = []
        self.executor = ScenarioExecutor()

    def add_scenario(self, scenario: ScenarioConfig, criteria: List[TestCriteria]):
        """Add a scenario to the test suite"""
        self.scenarios.append((scenario, criteria))

    def run(self) -> TestSuiteResult:
        """
        Execute all scenarios in the test suite

        Returns:
            Test suite result
        """
        print(f"\n{'#'*60}")
        print(f"# Running Test Suite: {self.suite_name}")
        print(f"# Total Scenarios: {len(self.scenarios)}")
        print(f"{'#'*60}\n")

        result = TestSuiteResult(
            suite_name=self.suite_name,
            start_time=time.time(),
            end_time=0.0
        )

        for scenario, criteria in self.scenarios:
            try:
                scenario_result = self.executor.execute_scenario(scenario, criteria)
                result.add_scenario_result(scenario_result)
            except Exception as e:
                print(f"ERROR executing scenario {scenario.scenario_id}: {e}")
                error_result = ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    scenario_type=scenario.scenario_type,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0.0,
                    result=TestResult.ERROR
                )
                error_result.add_issue(TestIssue(
                    timestamp=0.0,
                    severity=SeverityLevel.CRITICAL,
                    category="execution",
                    description=f"Execution error: {str(e)}"
                ))
                result.add_scenario_result(error_result)

        result.end_time = time.time()

        print(f"\n{'#'*60}")
        print(f"# Test Suite Complete: {self.suite_name}")
        print(f"# Pass Rate: {result.get_pass_rate():.1f}%")
        print(f"# Passed: {result.passed_scenarios}/{result.total_scenarios}")
        print(f"# Failed: {result.failed_scenarios}/{result.total_scenarios}")
        print(f"{'#'*60}\n")

        return result


class TestReportGenerator:
    """
    Test Report Generator

    Generates detailed test reports in various formats.
    """

    @staticmethod
    def generate_text_report(suite_result: TestSuiteResult) -> str:
        """Generate text-based test report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"TEST SUITE REPORT: {suite_result.suite_name}")
        lines.append("=" * 80)
        lines.append(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(suite_result.start_time))}")
        lines.append(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(suite_result.end_time))}")
        lines.append(f"Duration: {suite_result.end_time - suite_result.start_time:.2f}s")
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Scenarios: {suite_result.total_scenarios}")
        lines.append(f"Passed: {suite_result.passed_scenarios} ({suite_result.get_pass_rate():.1f}%)")
        lines.append(f"Failed: {suite_result.failed_scenarios}")
        lines.append(f"Warnings: {suite_result.warning_scenarios}")
        lines.append(f"Skipped: {suite_result.skipped_scenarios}")
        lines.append("")

        lines.append("SCENARIO RESULTS")
        lines.append("-" * 80)

        for idx, scenario_result in enumerate(suite_result.scenario_results, 1):
            status_symbol = "✓" if scenario_result.result == TestResult.PASSED else "✗"
            lines.append(f"\n{idx}. {status_symbol} {scenario_result.scenario_id} - {scenario_result.result.value.upper()}")
            lines.append(f"   Type: {scenario_result.scenario_type.value}")
            lines.append(f"   Duration: {scenario_result.duration:.2f}s")

            # Metrics
            if scenario_result.metrics:
                lines.append("   Metrics:")
                for metric in scenario_result.metrics:
                    status = "PASS" if metric.passed else "FAIL"
                    lines.append(f"     - {metric.name}: {metric.value:.2f} {metric.unit} [{status}]")

            # Issues
            if scenario_result.issues:
                lines.append("   Issues:")
                for issue in scenario_result.issues:
                    lines.append(f"     - [{issue.severity.value.upper()}] {issue.description}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    @staticmethod
    def generate_json_report(suite_result: TestSuiteResult) -> str:
        """Generate JSON test report"""
        report = {
            'suite_name': suite_result.suite_name,
            'start_time': suite_result.start_time,
            'end_time': suite_result.end_time,
            'summary': {
                'total': suite_result.total_scenarios,
                'passed': suite_result.passed_scenarios,
                'failed': suite_result.failed_scenarios,
                'warnings': suite_result.warning_scenarios,
                'skipped': suite_result.skipped_scenarios,
                'pass_rate': suite_result.get_pass_rate()
            },
            'scenarios': []
        }

        for scenario_result in suite_result.scenario_results:
            scenario_data = {
                'id': scenario_result.scenario_id,
                'type': scenario_result.scenario_type.value,
                'result': scenario_result.result.value,
                'duration': scenario_result.duration,
                'metrics': [
                    {
                        'name': m.name,
                        'type': m.metric_type.value,
                        'value': m.value,
                        'unit': m.unit,
                        'passed': m.passed
                    }
                    for m in scenario_result.metrics
                ],
                'issues': [
                    {
                        'timestamp': i.timestamp,
                        'severity': i.severity.value,
                        'category': i.category,
                        'description': i.description
                    }
                    for i in scenario_result.issues
                ]
            }
            report['scenarios'].append(scenario_data)

        return json.dumps(report, indent=2)


def main():
    """Test the scenario testing framework"""
    print("Testing Scenario Testing Framework\n")

    # Create test suite
    suite = TestSuite("Autonomous Driving Validation Suite")

    # Add predefined scenarios
    suite.add_scenario(*ScenarioLibrary.highway_cruise())
    suite.add_scenario(*ScenarioLibrary.emergency_braking())
    suite.add_scenario(*ScenarioLibrary.lane_change())
    suite.add_scenario(*ScenarioLibrary.cut_in())

    # Run test suite
    results = suite.run()

    # Generate reports
    print("\n" + TestReportGenerator.generate_text_report(results))

    # Save JSON report
    json_report = TestReportGenerator.generate_json_report(results)
    print(f"\nJSON report generated ({len(json_report)} bytes)")

    print("\n✓ Scenario Testing Framework test complete!")


if __name__ == "__main__":
    main()
