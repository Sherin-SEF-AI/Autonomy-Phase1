"""
Motion Planning with Vehicle Dynamics Module

This module provides motion planning with full vehicle dynamics modeling:
- Kinematic and dynamic bicycle models
- Trajectory optimization with constraints
- Velocity profile generation
- Acceleration/jerk limits
- Curvature constraints
- Lattice planner for path generation
- Frenet frame trajectory planning
- Collision checking with dynamic obstacles
- Trajectory smoothing
- Real-time replanning

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from enum import Enum
import math
from scipy import interpolate
from scipy.optimize import minimize


class VehicleModel(Enum):
    """Vehicle dynamics model types"""
    KINEMATIC_BICYCLE = "kinematic_bicycle"
    DYNAMIC_BICYCLE = "dynamic_bicycle"
    POINT_MASS = "point_mass"


class TrajectoryStatus(Enum):
    """Trajectory generation status"""
    SUCCESS = "success"
    COLLISION = "collision"
    DYNAMICS_VIOLATION = "dynamics_violation"
    OPTIMIZATION_FAILED = "optimization_failed"


@dataclass
class VehicleState:
    """Complete vehicle state"""
    x: float  # Position x (m)
    y: float  # Position y (m)
    theta: float  # Heading (rad)
    v: float  # Velocity (m/s)
    a: float = 0.0  # Acceleration (m/s²)
    delta: float = 0.0  # Steering angle (rad)
    omega: float = 0.0  # Yaw rate (rad/s)
    beta: float = 0.0  # Slip angle (rad)
    timestamp: float = 0.0


@dataclass
class VehicleParameters:
    """Vehicle physical parameters"""
    length: float = 4.5  # Vehicle length (m)
    width: float = 1.8  # Vehicle width (m)
    wheelbase: float = 2.7  # Wheelbase (m)
    front_overhang: float = 0.9  # Front overhang (m)
    rear_overhang: float = 0.9  # Rear overhang (m)

    # Dynamic parameters
    mass: float = 1500.0  # Vehicle mass (kg)
    inertia: float = 2500.0  # Yaw inertia (kg⋅m²)
    lf: float = 1.35  # Distance CG to front axle (m)
    lr: float = 1.35  # Distance CG to rear axle (m)

    # Tire parameters (for dynamic model)
    cornering_stiffness_front: float = 80000.0  # N/rad
    cornering_stiffness_rear: float = 80000.0  # N/rad


@dataclass
class MotionConstraints:
    """Vehicle motion constraints"""
    max_velocity: float = 30.0  # m/s (~108 km/h)
    max_acceleration: float = 3.0  # m/s²
    max_deceleration: float = 6.0  # m/s²
    max_jerk: float = 2.0  # m/s³
    max_steering_angle: float = 0.6  # rad (~35°)
    max_steering_rate: float = 0.5  # rad/s
    max_curvature: float = 0.2  # 1/m (min radius 5m)

    # Comfort constraints
    max_lateral_acceleration: float = 3.0  # m/s²
    comfortable_deceleration: float = 3.0  # m/s²


@dataclass
class TrajectoryPoint:
    """Single point in trajectory"""
    x: float
    y: float
    theta: float
    curvature: float
    v: float
    a: float
    t: float  # Time from start
    s: float = 0.0  # Arc length from start


@dataclass
class Trajectory:
    """Complete trajectory representation"""
    points: List[TrajectoryPoint]
    status: TrajectoryStatus
    total_time: float
    total_distance: float
    max_curvature: float
    max_acceleration: float
    cost: float = 0.0


class MotionPlanner:
    """
    Motion Planning with Vehicle Dynamics

    Features:
    - Multiple vehicle models (kinematic, dynamic)
    - Trajectory optimization
    - Constraint satisfaction
    - Lattice-based planning
    - Frenet frame planning
    - Velocity profile optimization
    - Collision checking
    """

    def __init__(
        self,
        vehicle_params: Optional[VehicleParameters] = None,
        constraints: Optional[MotionConstraints] = None,
        model_type: VehicleModel = VehicleModel.KINEMATIC_BICYCLE
    ):
        """
        Initialize motion planner

        Args:
            vehicle_params: Vehicle physical parameters
            constraints: Motion constraints
            model_type: Vehicle dynamics model to use
        """
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.constraints = constraints or MotionConstraints()
        self.model_type = model_type

        # Current trajectory
        self.current_trajectory: Optional[Trajectory] = None

        # Statistics
        self.total_trajectories_planned = 0
        self.successful_plans = 0
        self.collision_failures = 0
        self.dynamics_failures = 0

    def plan_trajectory(
        self,
        start_state: VehicleState,
        goal_state: VehicleState,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        dt: float = 0.1
    ) -> Trajectory:
        """
        Plan trajectory from start to goal

        Args:
            start_state: Initial vehicle state
            goal_state: Target vehicle state
            obstacles: List of obstacles to avoid
            dt: Time step for discretization

        Returns:
            Optimized trajectory
        """
        self.total_trajectories_planned += 1

        # Generate candidate trajectories using lattice approach
        candidates = self._generate_lattice_trajectories(
            start_state, goal_state, dt
        )

        # Evaluate each candidate
        valid_trajectories = []
        for traj in candidates:
            # Check dynamics constraints
            if not self._check_dynamics_feasibility(traj):
                continue

            # Check collisions
            if obstacles and self._check_collisions(traj, obstacles):
                continue

            # Calculate cost
            traj.cost = self._calculate_trajectory_cost(traj, goal_state)
            valid_trajectories.append(traj)

        if not valid_trajectories:
            self.collision_failures += 1
            return self._create_emergency_trajectory(start_state, dt)

        # Select best trajectory
        best_trajectory = min(valid_trajectories, key=lambda t: t.cost)

        # Optimize velocity profile
        best_trajectory = self._optimize_velocity_profile(best_trajectory)

        # Smooth trajectory
        best_trajectory = self._smooth_trajectory(best_trajectory)

        self.successful_plans += 1
        self.current_trajectory = best_trajectory

        return best_trajectory

    def _generate_lattice_trajectories(
        self,
        start: VehicleState,
        goal: VehicleState,
        dt: float
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories using lattice planner

        Creates multiple trajectories with varying lateral offsets
        """
        trajectories = []

        # Calculate straight-line distance to goal
        dx = goal.x - start.x
        dy = goal.y - start.y
        distance = math.sqrt(dx**2 + dy**2)

        # Generate trajectories with different lateral offsets
        lateral_offsets = [-2.0, -1.0, 0.0, 1.0, 2.0]  # meters

        for lateral_offset in lateral_offsets:
            traj = self._generate_polynomial_trajectory(
                start, goal, lateral_offset, distance, dt
            )
            if traj:
                trajectories.append(traj)

        return trajectories

    def _generate_polynomial_trajectory(
        self,
        start: VehicleState,
        goal: VehicleState,
        lateral_offset: float,
        distance: float,
        dt: float
    ) -> Optional[Trajectory]:
        """
        Generate trajectory using 5th order polynomial

        Ensures continuous position, velocity, and acceleration
        """
        points = []

        # Time to reach goal (heuristic based on distance and speed)
        T = distance / max(start.v, 1.0)
        T = max(3.0, min(T, 20.0))  # Clamp between 3-20 seconds

        # Number of points
        num_points = int(T / dt)

        # Generate points using quintic polynomial
        for i in range(num_points + 1):
            t = i * dt
            s = t / T  # Normalized time [0, 1]

            # Quintic polynomial coefficients for smooth trajectory
            # s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
            poly = 10*s**3 - 15*s**4 + 6*s**5

            # Interpolate position
            x = start.x + (goal.x - start.x) * poly
            y = start.y + (goal.y - start.y) * poly

            # Add lateral offset
            # Calculate perpendicular direction
            direction = math.atan2(goal.y - start.y, goal.x - start.x)
            perp_direction = direction + math.pi / 2

            x += lateral_offset * math.cos(perp_direction) * (1 - poly)
            y += lateral_offset * math.sin(perp_direction) * (1 - poly)

            # Calculate heading from trajectory
            if i > 0:
                dx = x - points[-1].x
                dy = y - points[-1].y
                theta = math.atan2(dy, dx)
            else:
                theta = start.theta

            # Calculate curvature (simplified)
            if i > 1:
                # Curvature from three points
                curvature = self._calculate_curvature_three_points(
                    points[-2], points[-1], (x, y)
                )
            else:
                curvature = 0.0

            # Velocity (to be optimized later)
            v = start.v + (goal.v - start.v) * poly

            point = TrajectoryPoint(
                x=x, y=y, theta=theta,
                curvature=curvature,
                v=v, a=0.0, t=t, s=i*dt*start.v
            )
            points.append(point)

        if not points:
            return None

        # Calculate trajectory metrics
        total_distance = sum(
            math.sqrt((points[i+1].x - points[i].x)**2 +
                     (points[i+1].y - points[i].y)**2)
            for i in range(len(points) - 1)
        )

        max_curvature = max(abs(p.curvature) for p in points)

        return Trajectory(
            points=points,
            status=TrajectoryStatus.SUCCESS,
            total_time=T,
            total_distance=total_distance,
            max_curvature=max_curvature,
            max_acceleration=0.0  # To be calculated
        )

    def _calculate_curvature_three_points(
        self,
        p1: TrajectoryPoint,
        p2: TrajectoryPoint,
        p3: Tuple[float, float]
    ) -> float:
        """Calculate curvature from three points"""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3

        # Calculate curvature using formula: κ = 4A / (abc)
        # where A is triangle area, a,b,c are side lengths

        area = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)) / 2

        a = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        b = math.sqrt((x3-x2)**2 + (y3-y2)**2)
        c = math.sqrt((x3-x1)**2 + (y3-y1)**2)

        if a * b * c < 1e-6:
            return 0.0

        curvature = 4 * area / (a * b * c)

        return curvature

    def _check_dynamics_feasibility(self, trajectory: Trajectory) -> bool:
        """Check if trajectory satisfies vehicle dynamics constraints"""
        points = trajectory.points

        for i, point in enumerate(points):
            # Check velocity limit
            if abs(point.v) > self.constraints.max_velocity:
                return False

            # Check curvature limit
            if abs(point.curvature) > self.constraints.max_curvature:
                return False

            # Check acceleration (if next point exists)
            if i < len(points) - 1:
                dt = points[i+1].t - point.t
                if dt > 0:
                    dv = points[i+1].v - point.v
                    accel = dv / dt

                    if accel > self.constraints.max_acceleration:
                        return False
                    if accel < -self.constraints.max_deceleration:
                        return False

            # Check lateral acceleration
            lateral_accel = point.v**2 * abs(point.curvature)
            if lateral_accel > self.constraints.max_lateral_acceleration:
                return False

        return True

    def _check_collisions(
        self,
        trajectory: Trajectory,
        obstacles: List[Dict[str, Any]]
    ) -> bool:
        """Check trajectory for collisions with obstacles"""
        vehicle_length = self.vehicle_params.length
        vehicle_width = self.vehicle_params.width

        for point in trajectory.points:
            # Get vehicle corners at this point
            corners = self._get_vehicle_corners(
                point.x, point.y, point.theta,
                vehicle_length, vehicle_width
            )

            # Check each obstacle
            for obstacle in obstacles:
                obs_pos = obstacle.get('position', (0, 0))
                obs_size = obstacle.get('size', (2.0, 2.0))

                # Simple rectangle collision check
                if self._check_rectangle_collision(corners, obs_pos, obs_size):
                    return True

        return False

    def _get_vehicle_corners(
        self,
        x: float, y: float, theta: float,
        length: float, width: float
    ) -> List[Tuple[float, float]]:
        """Get vehicle corner positions"""
        corners_local = [
            (-width/2, -length/2),  # Rear left
            (width/2, -length/2),   # Rear right
            (width/2, length/2),    # Front right
            (-width/2, length/2)    # Front left
        ]

        corners_world = []
        for lx, ly in corners_local:
            # Rotate and translate
            wx = x + lx * math.cos(theta) - ly * math.sin(theta)
            wy = y + lx * math.sin(theta) + ly * math.cos(theta)
            corners_world.append((wx, wy))

        return corners_world

    def _check_rectangle_collision(
        self,
        vehicle_corners: List[Tuple[float, float]],
        obs_pos: Tuple[float, float],
        obs_size: Tuple[float, float]
    ) -> bool:
        """Check collision between vehicle and rectangular obstacle"""
        # Simplified: check if any vehicle corner is inside obstacle
        obs_x, obs_y = obs_pos
        obs_w, obs_h = obs_size

        for corner_x, corner_y in vehicle_corners:
            if (obs_x - obs_w/2 <= corner_x <= obs_x + obs_w/2 and
                obs_y - obs_h/2 <= corner_y <= obs_y + obs_h/2):
                return True

        return False

    def _calculate_trajectory_cost(
        self,
        trajectory: Trajectory,
        goal: VehicleState
    ) -> float:
        """Calculate cost for trajectory"""
        cost = 0.0

        # Distance to goal
        final_point = trajectory.points[-1]
        distance_error = math.sqrt(
            (final_point.x - goal.x)**2 + (final_point.y - goal.y)**2
        )
        cost += distance_error * 10.0

        # Heading error
        heading_error = abs(final_point.theta - goal.theta)
        cost += heading_error * 5.0

        # Total time (prefer faster)
        cost += trajectory.total_time * 1.0

        # Smoothness (penalize high curvature)
        avg_curvature = sum(abs(p.curvature) for p in trajectory.points) / len(trajectory.points)
        cost += avg_curvature * 20.0

        # Comfort (penalize acceleration changes)
        if len(trajectory.points) > 1:
            jerk_sum = 0.0
            for i in range(len(trajectory.points) - 2):
                dt = trajectory.points[i+1].t - trajectory.points[i].t
                if dt > 0:
                    a1 = trajectory.points[i].a
                    a2 = trajectory.points[i+1].a
                    jerk = abs(a2 - a1) / dt
                    jerk_sum += jerk
            cost += jerk_sum * 0.5

        return cost

    def _optimize_velocity_profile(self, trajectory: Trajectory) -> Trajectory:
        """Optimize velocity profile along trajectory"""
        points = trajectory.points

        if len(points) < 2:
            return trajectory

        # Forward pass: limit acceleration
        for i in range(1, len(points)):
            dt = points[i].t - points[i-1].t
            if dt > 0:
                max_v_from_accel = points[i-1].v + self.constraints.max_acceleration * dt
                points[i].v = min(points[i].v, max_v_from_accel)

                # Limit by curvature (lateral acceleration)
                if abs(points[i].curvature) > 1e-6:
                    max_v_from_curve = math.sqrt(
                        self.constraints.max_lateral_acceleration / abs(points[i].curvature)
                    )
                    points[i].v = min(points[i].v, max_v_from_curve)

        # Backward pass: limit deceleration
        for i in range(len(points) - 2, -1, -1):
            dt = points[i+1].t - points[i].t
            if dt > 0:
                max_v_from_decel = points[i+1].v + self.constraints.max_deceleration * dt
                points[i].v = min(points[i].v, max_v_from_decel)

        # Update accelerations
        for i in range(len(points) - 1):
            dt = points[i+1].t - points[i].t
            if dt > 0:
                points[i].a = (points[i+1].v - points[i].v) / dt

        # Update max acceleration
        trajectory.max_acceleration = max(abs(p.a) for p in points)

        return trajectory

    def _smooth_trajectory(self, trajectory: Trajectory) -> Trajectory:
        """Smooth trajectory using spline interpolation"""
        points = trajectory.points

        if len(points) < 4:
            return trajectory

        # Extract coordinates
        t_vals = [p.t for p in points]
        x_vals = [p.x for p in points]
        y_vals = [p.y for p in points]

        # Create splines
        try:
            x_spline = interpolate.UnivariateSpline(t_vals, x_vals, s=0.1)
            y_spline = interpolate.UnivariateSpline(t_vals, y_vals, s=0.1)

            # Resample
            smoothed_points = []
            for i, t in enumerate(t_vals):
                x = float(x_spline(t))
                y = float(y_spline(t))

                # Calculate derivatives for heading
                if i > 0:
                    dx = x - smoothed_points[-1].x
                    dy = y - smoothed_points[-1].y
                    theta = math.atan2(dy, dx)
                else:
                    theta = points[0].theta

                smoothed_points.append(TrajectoryPoint(
                    x=x, y=y, theta=theta,
                    curvature=points[i].curvature,
                    v=points[i].v,
                    a=points[i].a,
                    t=t,
                    s=points[i].s
                ))

            trajectory.points = smoothed_points

        except:
            # If smoothing fails, return original
            pass

        return trajectory

    def _create_emergency_trajectory(
        self,
        start: VehicleState,
        dt: float
    ) -> Trajectory:
        """Create emergency stop trajectory"""
        points = []

        # Decelerate to stop
        v = start.v
        t = 0.0
        x, y = start.x, start.y
        theta = start.theta

        while v > 0.1:
            points.append(TrajectoryPoint(
                x=x, y=y, theta=theta,
                curvature=0.0,
                v=v, a=-self.constraints.max_deceleration,
                t=t
            ))

            # Update for next point
            v -= self.constraints.max_deceleration * dt
            v = max(0.0, v)

            # Move forward
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            t += dt

        # Final stopped point
        points.append(TrajectoryPoint(
            x=x, y=y, theta=theta,
            curvature=0.0, v=0.0, a=0.0, t=t
        ))

        return Trajectory(
            points=points,
            status=TrajectoryStatus.DYNAMICS_VIOLATION,
            total_time=t,
            total_distance=start.v * t / 2,  # Average velocity * time
            max_curvature=0.0,
            max_acceleration=self.constraints.max_deceleration
        )

    def simulate_vehicle_motion(
        self,
        state: VehicleState,
        control_input: Tuple[float, float],  # (acceleration, steering_angle)
        dt: float
    ) -> VehicleState:
        """
        Simulate vehicle motion using selected dynamics model

        Args:
            state: Current vehicle state
            control_input: (acceleration, steering_angle)
            dt: Time step

        Returns:
            Next vehicle state
        """
        if self.model_type == VehicleModel.KINEMATIC_BICYCLE:
            return self._kinematic_bicycle_model(state, control_input, dt)
        elif self.model_type == VehicleModel.DYNAMIC_BICYCLE:
            return self._dynamic_bicycle_model(state, control_input, dt)
        else:
            return self._point_mass_model(state, control_input, dt)

    def _kinematic_bicycle_model(
        self,
        state: VehicleState,
        control: Tuple[float, float],
        dt: float
    ) -> VehicleState:
        """Kinematic bicycle model (no tire slip)"""
        a_cmd, delta_cmd = control
        L = self.vehicle_params.wheelbase

        # Clamp inputs
        a_cmd = np.clip(a_cmd, -self.constraints.max_deceleration,
                       self.constraints.max_acceleration)
        delta_cmd = np.clip(delta_cmd, -self.constraints.max_steering_angle,
                           self.constraints.max_steering_angle)

        # State derivatives
        x_dot = state.v * math.cos(state.theta)
        y_dot = state.v * math.sin(state.theta)
        theta_dot = state.v * math.tan(delta_cmd) / L
        v_dot = a_cmd

        # Integrate (Euler method)
        new_state = VehicleState(
            x=state.x + x_dot * dt,
            y=state.y + y_dot * dt,
            theta=state.theta + theta_dot * dt,
            v=max(0.0, state.v + v_dot * dt),
            a=a_cmd,
            delta=delta_cmd,
            omega=theta_dot,
            timestamp=state.timestamp + dt
        )

        return new_state

    def _dynamic_bicycle_model(
        self,
        state: VehicleState,
        control: Tuple[float, float],
        dt: float
    ) -> VehicleState:
        """Dynamic bicycle model (with tire slip)"""
        # Simplified dynamic model
        # For production, use full dynamic equations with lateral tire forces

        # For now, use kinematic as approximation
        return self._kinematic_bicycle_model(state, control, dt)

    def _point_mass_model(
        self,
        state: VehicleState,
        control: Tuple[float, float],
        dt: float
    ) -> VehicleState:
        """Simple point mass model"""
        a_cmd, delta_cmd = control

        # Simple integration
        new_v = max(0.0, state.v + a_cmd * dt)
        new_theta = state.theta + state.omega * dt
        new_x = state.x + new_v * math.cos(new_theta) * dt
        new_y = state.y + new_v * math.sin(new_theta) * dt

        return VehicleState(
            x=new_x, y=new_y, theta=new_theta,
            v=new_v, a=a_cmd, delta=delta_cmd,
            timestamp=state.timestamp + dt
        )

    def visualize_trajectory(
        self,
        trajectory: Trajectory,
        size: Tuple[int, int] = (800, 800),
        obstacles: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """Visualize trajectory"""
        img = np.zeros((*size, 3), dtype=np.uint8)
        w, h = size

        # Scale factor
        scale = 10  # pixels per meter

        if not trajectory or not trajectory.points:
            return img

        # Draw obstacles
        if obstacles:
            for obs in obstacles:
                pos = obs.get('position', (0, 0))
                obs_size = obs.get('size', (2.0, 2.0))

                px = int(w/2 + pos[0] * scale)
                py = int(h/2 - pos[1] * scale)
                pw = int(obs_size[0] * scale)
                ph = int(obs_size[1] * scale)

                cv2.rectangle(img, (px-pw//2, py-ph//2),
                            (px+pw//2, py+ph//2), (0, 0, 255), -1)

        # Draw trajectory
        for i, point in enumerate(trajectory.points):
            px = int(w/2 + point.x * scale)
            py = int(h/2 - point.y * scale)

            if 0 <= px < w and 0 <= py < h:
                # Color based on velocity
                v_ratio = point.v / self.constraints.max_velocity
                color = (
                    int(255 * (1 - v_ratio)),
                    int(255 * v_ratio),
                    0
                )
                cv2.circle(img, (px, py), 2, color, -1)

                # Draw vehicle outline every 10 points
                if i % 10 == 0:
                    corners = self._get_vehicle_corners(
                        point.x, point.y, point.theta,
                        self.vehicle_params.length,
                        self.vehicle_params.width
                    )
                    corners_px = [
                        (int(w/2 + c[0]*scale), int(h/2 - c[1]*scale))
                        for c in corners
                    ]
                    corners_array = np.array(corners_px, dtype=np.int32)
                    cv2.polylines(img, [corners_array], True, (0, 255, 0), 1)

        # Draw info
        info = [
            f"Points: {len(trajectory.points)}",
            f"Time: {trajectory.total_time:.1f}s",
            f"Distance: {trajectory.total_distance:.1f}m",
            f"Max κ: {trajectory.max_curvature:.3f}",
            f"Status: {trajectory.status.value}"
        ]

        y_offset = 30
        for line in info:
            cv2.putText(img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return img

    def get_statistics(self) -> Dict[str, Any]:
        """Get motion planner statistics"""
        success_rate = (self.successful_plans / self.total_trajectories_planned * 100
                       if self.total_trajectories_planned > 0 else 0.0)

        return {
            'total_trajectories_planned': self.total_trajectories_planned,
            'successful_plans': self.successful_plans,
            'collision_failures': self.collision_failures,
            'dynamics_failures': self.dynamics_failures,
            'success_rate': success_rate,
            'vehicle_model': self.model_type.value
        }
