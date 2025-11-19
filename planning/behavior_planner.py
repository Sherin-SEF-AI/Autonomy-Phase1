"""
Behavior Planning & Decision Making Module

This module provides high-level behavior planning and decision making:
- Finite State Machine for driving behaviors
- Strategic decision making (lane change, overtake, stop, yield)
- Traffic rule compliance
- Intersection handling
- Emergency behavior arbitration
- Cost-based behavior selection
- Goal-driven planning
- Situation assessment

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from collections import deque
import time


class DrivingBehavior(Enum):
    """High-level driving behaviors"""
    IDLE = "idle"
    CRUISE = "cruise"
    FOLLOW = "follow"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    OVERTAKE = "overtake"
    MERGE = "merge"
    YIELD = "yield"
    STOP = "stop"
    PARK = "park"
    EMERGENCY_STOP = "emergency_stop"
    INTERSECTION_CROSS = "intersection_cross"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class BehaviorState(Enum):
    """Behavior execution states"""
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class DecisionFactor(Enum):
    """Factors influencing decisions"""
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    COMFORT = "comfort"
    LEGALITY = "legality"
    PROGRESS = "progress"


@dataclass
class BehaviorCost:
    """Cost components for behavior evaluation"""
    safety_cost: float = 0.0  # Collision risk, TTC violations
    efficiency_cost: float = 0.0  # Time, fuel consumption
    comfort_cost: float = 0.0  # Acceleration, jerk
    legality_cost: float = 0.0  # Traffic rule violations
    progress_cost: float = 0.0  # Distance to goal

    def get_total_cost(self, weights: Optional[Dict[DecisionFactor, float]] = None) -> float:
        """Calculate weighted total cost"""
        if weights is None:
            weights = {
                DecisionFactor.SAFETY: 10.0,
                DecisionFactor.EFFICIENCY: 1.0,
                DecisionFactor.COMFORT: 2.0,
                DecisionFactor.LEGALITY: 8.0,
                DecisionFactor.PROGRESS: 1.5
            }

        total = (
            self.safety_cost * weights[DecisionFactor.SAFETY] +
            self.efficiency_cost * weights[DecisionFactor.EFFICIENCY] +
            self.comfort_cost * weights[DecisionFactor.COMFORT] +
            self.legality_cost * weights[DecisionFactor.LEGALITY] +
            self.progress_cost * weights[DecisionFactor.PROGRESS]
        )
        return total


@dataclass
class BehaviorCandidate:
    """Candidate behavior with cost and feasibility"""
    behavior: DrivingBehavior
    cost: BehaviorCost
    is_feasible: bool
    confidence: float  # 0-1
    duration_estimate: float  # seconds
    reason: str = ""


@dataclass
class DrivingSituation:
    """Current driving situation assessment"""
    # Ego state
    ego_speed: float  # m/s
    ego_position: Tuple[float, float, float]
    ego_lane_id: Optional[int]

    # Traffic
    lead_vehicle_distance: Optional[float]  # meters
    lead_vehicle_speed: Optional[float]  # m/s
    adjacent_left_vehicle: bool
    adjacent_right_vehicle: bool

    # Environment
    speed_limit: Optional[float]  # m/s
    in_intersection: bool
    approaching_intersection: bool
    traffic_light_state: Optional[str]  # "red", "yellow", "green"
    stop_line_distance: Optional[float]  # meters

    # Goal
    goal_lane_id: Optional[int]
    distance_to_goal: Optional[float]  # meters

    # Safety
    collision_warnings: List[Any] = field(default_factory=list)
    emergency_situation: bool = False


@dataclass
class BehaviorPlan:
    """Planned behavior sequence"""
    primary_behavior: DrivingBehavior
    fallback_behaviors: List[DrivingBehavior]
    state: BehaviorState
    start_time: float
    expected_duration: float
    cost: BehaviorCost
    constraints: Dict[str, Any] = field(default_factory=dict)


class BehaviorPlanner:
    """
    High-Level Behavior Planner

    Features:
    - Finite State Machine for behavior sequencing
    - Cost-based behavior selection
    - Multi-criteria decision making
    - Traffic rule compliance
    - Emergency behavior handling
    - Situation-aware planning
    """

    def __init__(self):
        """Initialize behavior planner"""
        self.current_behavior: DrivingBehavior = DrivingBehavior.IDLE
        self.current_plan: Optional[BehaviorPlan] = None
        self.behavior_history: deque = deque(maxlen=50)

        # Decision weights (can be tuned)
        self.decision_weights = {
            DecisionFactor.SAFETY: 10.0,
            DecisionFactor.EFFICIENCY: 1.0,
            DecisionFactor.COMFORT: 2.0,
            DecisionFactor.LEGALITY: 8.0,
            DecisionFactor.PROGRESS: 1.5
        }

        # Behavior transition rules (FSM)
        self.transition_rules = self._define_transition_rules()

        # Statistics
        self.total_decisions = 0
        self.behavior_counts: Dict[DrivingBehavior, int] = {}
        self.emergency_stops = 0

    def _define_transition_rules(self) -> Dict[DrivingBehavior, Set[DrivingBehavior]]:
        """Define allowed behavior transitions (FSM)"""
        return {
            DrivingBehavior.IDLE: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP,
                DrivingBehavior.PARK
            },
            DrivingBehavior.CRUISE: {
                DrivingBehavior.FOLLOW,
                DrivingBehavior.LANE_CHANGE_LEFT,
                DrivingBehavior.LANE_CHANGE_RIGHT,
                DrivingBehavior.OVERTAKE,
                DrivingBehavior.STOP,
                DrivingBehavior.YIELD,
                DrivingBehavior.INTERSECTION_CROSS,
                DrivingBehavior.TURN_LEFT,
                DrivingBehavior.TURN_RIGHT,
                DrivingBehavior.EMERGENCY_STOP
            },
            DrivingBehavior.FOLLOW: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.LANE_CHANGE_LEFT,
                DrivingBehavior.LANE_CHANGE_RIGHT,
                DrivingBehavior.OVERTAKE,
                DrivingBehavior.STOP,
                DrivingBehavior.EMERGENCY_STOP
            },
            DrivingBehavior.LANE_CHANGE_LEFT: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.FOLLOW,
                DrivingBehavior.STOP
            },
            DrivingBehavior.LANE_CHANGE_RIGHT: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.FOLLOW,
                DrivingBehavior.STOP
            },
            DrivingBehavior.OVERTAKE: {
                DrivingBehavior.LANE_CHANGE_RIGHT,
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.STOP: {
                DrivingBehavior.IDLE,
                DrivingBehavior.CRUISE,
                DrivingBehavior.PARK
            },
            DrivingBehavior.YIELD: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.INTERSECTION_CROSS: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.TURN_LEFT: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.TURN_RIGHT: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.EMERGENCY_STOP: {
                DrivingBehavior.IDLE,
                DrivingBehavior.STOP
            },
            DrivingBehavior.PARK: {
                DrivingBehavior.IDLE
            },
            DrivingBehavior.MERGE: {
                DrivingBehavior.CRUISE,
                DrivingBehavior.STOP
            }
        }

    def plan_behavior(
        self,
        situation: DrivingSituation,
        timestamp: float
    ) -> BehaviorPlan:
        """
        Plan next behavior based on current situation

        Args:
            situation: Current driving situation
            timestamp: Current timestamp

        Returns:
            Behavior plan with primary and fallback behaviors
        """
        self.total_decisions += 1

        # Emergency handling (highest priority)
        if situation.emergency_situation or any(
            w.risk_level == "critical" or w.risk_level == "imminent"
            for w in situation.collision_warnings
        ):
            return self._create_emergency_plan(timestamp)

        # Generate behavior candidates
        candidates = self._generate_behavior_candidates(situation)

        # Evaluate costs for each candidate
        for candidate in candidates:
            candidate.cost = self._evaluate_behavior_cost(candidate, situation)

        # Filter feasible behaviors
        feasible_candidates = [c for c in candidates if c.is_feasible]

        if not feasible_candidates:
            # No feasible behaviors - default to stop
            return self._create_stop_plan(timestamp, reason="No feasible behaviors")

        # Select best behavior based on cost
        best_candidate = min(
            feasible_candidates,
            key=lambda c: c.cost.get_total_cost(self.decision_weights)
        )

        # Create plan with fallbacks
        fallbacks = self._select_fallback_behaviors(feasible_candidates, best_candidate)

        plan = BehaviorPlan(
            primary_behavior=best_candidate.behavior,
            fallback_behaviors=fallbacks,
            state=BehaviorState.PLANNING,
            start_time=timestamp,
            expected_duration=best_candidate.duration_estimate,
            cost=best_candidate.cost
        )

        # Update state
        self.current_behavior = best_candidate.behavior
        self.current_plan = plan
        self.behavior_history.append((timestamp, best_candidate.behavior))

        # Update statistics
        self.behavior_counts[best_candidate.behavior] = \
            self.behavior_counts.get(best_candidate.behavior, 0) + 1

        return plan

    def _generate_behavior_candidates(
        self,
        situation: DrivingSituation
    ) -> List[BehaviorCandidate]:
        """Generate possible behavior candidates"""
        candidates = []

        # Get allowed transitions from current behavior
        allowed_behaviors = self.transition_rules.get(self.current_behavior, set())

        # Always allow emergency stop
        allowed_behaviors.add(DrivingBehavior.EMERGENCY_STOP)

        # Generate candidates based on situation
        for behavior in allowed_behaviors:
            candidate = self._create_behavior_candidate(behavior, situation)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _create_behavior_candidate(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> Optional[BehaviorCandidate]:
        """Create behavior candidate with feasibility check"""
        # Check basic feasibility
        is_feasible, reason = self._check_feasibility(behavior, situation)

        # Estimate duration
        duration = self._estimate_behavior_duration(behavior, situation)

        return BehaviorCandidate(
            behavior=behavior,
            cost=BehaviorCost(),  # Will be computed later
            is_feasible=is_feasible,
            confidence=0.9 if is_feasible else 0.1,
            duration_estimate=duration,
            reason=reason
        )

    def _check_feasibility(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> Tuple[bool, str]:
        """Check if behavior is feasible in current situation"""
        if behavior == DrivingBehavior.LANE_CHANGE_LEFT:
            if situation.adjacent_left_vehicle:
                return False, "Vehicle in left lane"
            if situation.ego_lane_id is None:
                return False, "No current lane"
            return True, "Feasible"

        elif behavior == DrivingBehavior.LANE_CHANGE_RIGHT:
            if situation.adjacent_right_vehicle:
                return False, "Vehicle in right lane"
            if situation.ego_lane_id is None:
                return False, "No current lane"
            return True, "Feasible"

        elif behavior == DrivingBehavior.OVERTAKE:
            if situation.lead_vehicle_distance is None:
                return False, "No lead vehicle"
            if situation.adjacent_left_vehicle:
                return False, "Cannot change to left lane"
            return True, "Feasible"

        elif behavior == DrivingBehavior.FOLLOW:
            if situation.lead_vehicle_distance is None:
                return False, "No vehicle to follow"
            if situation.lead_vehicle_distance > 50.0:
                return False, "Lead vehicle too far"
            return True, "Feasible"

        elif behavior == DrivingBehavior.INTERSECTION_CROSS:
            if not situation.in_intersection and not situation.approaching_intersection:
                return False, "No intersection"
            if situation.traffic_light_state == "red":
                return False, "Red light"
            return True, "Feasible"

        elif behavior == DrivingBehavior.STOP:
            return True, "Always feasible"

        elif behavior == DrivingBehavior.CRUISE:
            return True, "Feasible"

        elif behavior == DrivingBehavior.EMERGENCY_STOP:
            return True, "Always feasible"

        # Default
        return True, "Feasible"

    def _evaluate_behavior_cost(
        self,
        candidate: BehaviorCandidate,
        situation: DrivingSituation
    ) -> BehaviorCost:
        """Evaluate cost for behavior candidate"""
        cost = BehaviorCost()

        behavior = candidate.behavior

        # Safety cost
        cost.safety_cost = self._calculate_safety_cost(behavior, situation)

        # Efficiency cost
        cost.efficiency_cost = self._calculate_efficiency_cost(behavior, situation)

        # Comfort cost
        cost.comfort_cost = self._calculate_comfort_cost(behavior, situation)

        # Legality cost
        cost.legality_cost = self._calculate_legality_cost(behavior, situation)

        # Progress cost
        cost.progress_cost = self._calculate_progress_cost(behavior, situation)

        return cost

    def _calculate_safety_cost(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Calculate safety cost component"""
        cost = 0.0

        # Collision warnings
        if situation.collision_warnings:
            max_risk = max(
                1.0 if w.risk_level in ["critical", "imminent"] else
                0.7 if w.risk_level == "high" else
                0.3 if w.risk_level == "medium" else 0.1
                for w in situation.collision_warnings
            )
            cost += max_risk * 100.0

        # Lead vehicle distance
        if situation.lead_vehicle_distance is not None:
            if situation.lead_vehicle_distance < 10.0:
                cost += (10.0 - situation.lead_vehicle_distance) * 5.0

        # Lane changes are inherently riskier
        if behavior in [DrivingBehavior.LANE_CHANGE_LEFT, DrivingBehavior.LANE_CHANGE_RIGHT]:
            cost += 10.0

        # Overtaking is risky
        if behavior == DrivingBehavior.OVERTAKE:
            cost += 20.0

        return cost

    def _calculate_efficiency_cost(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Calculate efficiency cost (time, fuel)"""
        cost = 0.0

        # Stopping is inefficient
        if behavior == DrivingBehavior.STOP:
            cost += 50.0

        # Following slow vehicle is inefficient
        if behavior == DrivingBehavior.FOLLOW:
            if situation.lead_vehicle_speed is not None:
                if situation.lead_vehicle_speed < situation.ego_speed * 0.8:
                    cost += 20.0

        # Lane changes take time
        if behavior in [DrivingBehavior.LANE_CHANGE_LEFT, DrivingBehavior.LANE_CHANGE_RIGHT]:
            cost += 10.0

        return cost

    def _calculate_comfort_cost(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Calculate comfort cost (acceleration, jerk)"""
        cost = 0.0

        # Emergency stop is uncomfortable
        if behavior == DrivingBehavior.EMERGENCY_STOP:
            cost += 100.0

        # Hard stops are uncomfortable
        if behavior == DrivingBehavior.STOP:
            if situation.ego_speed > 10.0:  # High speed stop
                cost += 30.0

        # Lane changes require lateral acceleration
        if behavior in [DrivingBehavior.LANE_CHANGE_LEFT, DrivingBehavior.LANE_CHANGE_RIGHT]:
            cost += 15.0

        return cost

    def _calculate_legality_cost(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Calculate legality cost (traffic rule violations)"""
        cost = 0.0

        # Running red light
        if behavior in [DrivingBehavior.CRUISE, DrivingBehavior.INTERSECTION_CROSS]:
            if situation.traffic_light_state == "red":
                cost += 1000.0  # Extremely high cost

        # Speeding
        if behavior == DrivingBehavior.CRUISE:
            if situation.speed_limit and situation.ego_speed > situation.speed_limit:
                cost += (situation.ego_speed - situation.speed_limit) * 20.0

        # Not stopping at stop line
        if situation.stop_line_distance is not None and situation.stop_line_distance < 1.0:
            if behavior != DrivingBehavior.STOP:
                cost += 500.0

        return cost

    def _calculate_progress_cost(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Calculate progress cost (distance to goal)"""
        cost = 0.0

        # Stopping reduces progress
        if behavior == DrivingBehavior.STOP:
            cost += 40.0

        # Not being in goal lane
        if situation.goal_lane_id is not None and situation.ego_lane_id is not None:
            if situation.ego_lane_id != situation.goal_lane_id:
                cost += 30.0

                # Reward lane changes toward goal
                # (would need more context to determine which direction)

        return cost

    def _estimate_behavior_duration(
        self,
        behavior: DrivingBehavior,
        situation: DrivingSituation
    ) -> float:
        """Estimate duration of behavior in seconds"""
        if behavior == DrivingBehavior.LANE_CHANGE_LEFT:
            return 5.0
        elif behavior == DrivingBehavior.LANE_CHANGE_RIGHT:
            return 5.0
        elif behavior == DrivingBehavior.OVERTAKE:
            return 10.0
        elif behavior == DrivingBehavior.STOP:
            return 3.0
        elif behavior == DrivingBehavior.EMERGENCY_STOP:
            return 2.0
        elif behavior == DrivingBehavior.INTERSECTION_CROSS:
            return 4.0
        elif behavior == DrivingBehavior.TURN_LEFT:
            return 6.0
        elif behavior == DrivingBehavior.TURN_RIGHT:
            return 5.0
        else:
            return 1.0  # Default for continuous behaviors

    def _select_fallback_behaviors(
        self,
        candidates: List[BehaviorCandidate],
        primary: BehaviorCandidate
    ) -> List[DrivingBehavior]:
        """Select fallback behaviors"""
        # Sort by cost
        sorted_candidates = sorted(
            [c for c in candidates if c.behavior != primary.behavior],
            key=lambda c: c.cost.get_total_cost(self.decision_weights)
        )

        # Return top 2 fallbacks
        return [c.behavior for c in sorted_candidates[:2]]

    def _create_emergency_plan(self, timestamp: float) -> BehaviorPlan:
        """Create emergency stop plan"""
        self.emergency_stops += 1

        return BehaviorPlan(
            primary_behavior=DrivingBehavior.EMERGENCY_STOP,
            fallback_behaviors=[DrivingBehavior.STOP],
            state=BehaviorState.EXECUTING,
            start_time=timestamp,
            expected_duration=2.0,
            cost=BehaviorCost(safety_cost=0.0),  # Safety is paramount
            constraints={'max_deceleration': 8.0}
        )

    def _create_stop_plan(self, timestamp: float, reason: str) -> BehaviorPlan:
        """Create stop behavior plan"""
        return BehaviorPlan(
            primary_behavior=DrivingBehavior.STOP,
            fallback_behaviors=[DrivingBehavior.IDLE],
            state=BehaviorState.EXECUTING,
            start_time=timestamp,
            expected_duration=3.0,
            cost=BehaviorCost(),
            constraints={'reason': reason}
        )

    def get_current_behavior(self) -> DrivingBehavior:
        """Get current active behavior"""
        return self.current_behavior

    def update_behavior_state(self, new_state: BehaviorState):
        """Update state of current behavior"""
        if self.current_plan:
            self.current_plan.state = new_state

    def get_statistics(self) -> Dict[str, Any]:
        """Get behavior planner statistics"""
        most_common_behavior = max(
            self.behavior_counts.items(),
            key=lambda x: x[1],
            default=(DrivingBehavior.IDLE, 0)
        )

        return {
            'total_decisions': self.total_decisions,
            'current_behavior': self.current_behavior.value,
            'emergency_stops': self.emergency_stops,
            'behavior_counts': {k.value: v for k, v in self.behavior_counts.items()},
            'most_common_behavior': most_common_behavior[0].value,
            'behavior_history_length': len(self.behavior_history)
        }
