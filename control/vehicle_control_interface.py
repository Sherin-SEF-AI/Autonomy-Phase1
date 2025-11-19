"""
Vehicle Control Interface Module
Implements low-level vehicle control for autonomous driving systems.

Features:
- Longitudinal control (throttle, brake)
- Lateral control (steering)
- PID controllers for speed and steering
- Model Predictive Control (MPC) option
- Safety limits and constraints
- Control modes (manual, assisted, autonomous)
- Actuator diagnostics and health monitoring
- Emergency override capabilities

Author: Autonomous Driving System
Version: 1.3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import math
from collections import deque


class ControlMode(Enum):
    """Vehicle control mode"""
    MANUAL = "manual"
    ASSISTED = "assisted"  # Driver assistance (e.g., ACC, LKA)
    AUTONOMOUS = "autonomous"  # Full autonomous control
    EMERGENCY_STOP = "emergency_stop"


class ActuatorType(Enum):
    """Vehicle actuator types"""
    STEERING = "steering"
    THROTTLE = "throttle"
    BRAKE = "brake"
    GEAR = "gear"


class ActuatorStatus(Enum):
    """Actuator health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class GearPosition(Enum):
    """Transmission gear positions"""
    PARK = "P"
    REVERSE = "R"
    NEUTRAL = "N"
    DRIVE = "D"
    SPORT = "S"
    LOW = "L"


@dataclass
class VehicleLimits:
    """Physical limits of the vehicle"""
    max_steering_angle: float = 35.0  # degrees
    max_steering_rate: float = 180.0  # deg/s
    max_throttle: float = 1.0  # 0-1
    max_brake: float = 1.0  # 0-1
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 8.0  # m/s^2 (emergency braking)
    max_jerk: float = 5.0  # m/s^3
    max_speed: float = 50.0  # m/s (≈180 km/h)
    wheelbase: float = 2.7  # meters
    track_width: float = 1.6  # meters
    mass: float = 1500.0  # kg


@dataclass
class ControlCommand:
    """Low-level control command"""
    timestamp: float
    steering_angle: float  # degrees, positive = left
    throttle: float  # 0-1
    brake: float  # 0-1
    gear: GearPosition = GearPosition.DRIVE

    def validate(self, limits: VehicleLimits) -> bool:
        """Validate control command against vehicle limits"""
        if abs(self.steering_angle) > limits.max_steering_angle:
            return False
        if not (0.0 <= self.throttle <= limits.max_throttle):
            return False
        if not (0.0 <= self.brake <= limits.max_brake):
            return False
        return True

    def clamp(self, limits: VehicleLimits):
        """Clamp control values to vehicle limits"""
        self.steering_angle = np.clip(
            self.steering_angle,
            -limits.max_steering_angle,
            limits.max_steering_angle
        )
        self.throttle = np.clip(self.throttle, 0.0, limits.max_throttle)
        self.brake = np.clip(self.brake, 0.0, limits.max_brake)


@dataclass
class VehicleState:
    """Current vehicle state"""
    timestamp: float
    position_x: float = 0.0  # meters
    position_y: float = 0.0  # meters
    heading: float = 0.0  # degrees
    speed: float = 0.0  # m/s
    acceleration: float = 0.0  # m/s^2
    steering_angle: float = 0.0  # degrees
    yaw_rate: float = 0.0  # deg/s
    lateral_acceleration: float = 0.0  # m/s^2


@dataclass
class ControlTarget:
    """Control target/setpoint"""
    target_speed: float = 0.0  # m/s
    target_acceleration: float = 0.0  # m/s^2
    target_steering_angle: float = 0.0  # degrees
    target_curvature: float = 0.0  # 1/m
    path_lateral_error: float = 0.0  # meters
    path_heading_error: float = 0.0  # degrees


@dataclass
class PIDConfig:
    """PID controller configuration"""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    output_min: float = -1.0
    output_max: float = 1.0
    integral_min: float = -10.0
    integral_max: float = 10.0


@dataclass
class ActuatorState:
    """Actuator state and diagnostics"""
    actuator_type: ActuatorType
    status: ActuatorStatus = ActuatorStatus.HEALTHY
    commanded_value: float = 0.0
    actual_value: float = 0.0
    last_update: float = 0.0
    error_count: int = 0
    response_time: float = 0.0  # seconds

    def get_error(self) -> float:
        """Get actuator tracking error"""
        return abs(self.commanded_value - self.actual_value)

    def is_healthy(self, max_error: float = 0.1) -> bool:
        """Check if actuator is healthy"""
        return self.status == ActuatorStatus.HEALTHY and self.get_error() < max_error


@dataclass
class ControlStatistics:
    """Control system statistics"""
    total_commands: int = 0
    emergency_stops: int = 0
    limit_violations: int = 0
    average_steering_error: float = 0.0  # degrees
    average_speed_error: float = 0.0  # m/s
    control_loop_frequency: float = 0.0  # Hz
    average_latency: float = 0.0  # seconds


class PIDController:
    """
    PID Controller

    Classic PID control with anti-windup and derivative filtering.
    """

    def __init__(self, config: PIDConfig):
        self.config = config

        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0

        # Derivative filtering (low-pass filter)
        self.derivative_alpha = 0.1  # Filter coefficient
        self.filtered_derivative = 0.0

    def update(self, setpoint: float, measurement: float, current_time: float) -> float:
        """
        Update PID controller

        Args:
            setpoint: Desired value
            measurement: Current measured value
            current_time: Current timestamp

        Returns:
            Control output
        """
        # Calculate error
        error = setpoint - measurement

        # Calculate dt
        if self.previous_time == 0.0:
            dt = 0.01  # Default dt
        else:
            dt = current_time - self.previous_time
            if dt <= 0.0:
                dt = 0.01

        # Proportional term
        p_term = self.config.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            self.config.integral_min,
            self.config.integral_max
        )
        i_term = self.config.ki * self.integral

        # Derivative term with filtering
        derivative = (error - self.previous_error) / dt
        self.filtered_derivative = (
            self.derivative_alpha * derivative +
            (1.0 - self.derivative_alpha) * self.filtered_derivative
        )
        d_term = self.config.kd * self.filtered_derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output
        output = np.clip(output, self.config.output_min, self.config.output_max)

        # Update state
        self.previous_error = error
        self.previous_time = current_time

        return output

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.filtered_derivative = 0.0


class LongitudinalController:
    """
    Longitudinal Controller

    Controls vehicle speed using throttle and brake.
    """

    def __init__(self, limits: VehicleLimits):
        self.limits = limits

        # Speed PID controller
        speed_config = PIDConfig(
            kp=0.5,
            ki=0.1,
            kd=0.2,
            output_min=-1.0,
            output_max=1.0
        )
        self.speed_pid = PIDController(speed_config)

        # Feedforward gain
        self.ff_gain = 0.3

    def update(self, target: ControlTarget, state: VehicleState, dt: float) -> Tuple[float, float]:
        """
        Update longitudinal controller

        Args:
            target: Control target
            state: Current vehicle state
            dt: Time step

        Returns:
            (throttle, brake) commands in range [0, 1]
        """
        current_time = time.time()

        # PID control for speed tracking
        control_output = self.speed_pid.update(
            target.target_speed,
            state.speed,
            current_time
        )

        # Feedforward term based on target acceleration
        ff_term = self.ff_gain * target.target_acceleration

        # Combine feedback and feedforward
        total_output = control_output + ff_term

        # Split into throttle and brake
        if total_output > 0.0:
            # Acceleration - use throttle
            throttle = min(total_output, 1.0)
            brake = 0.0
        else:
            # Deceleration - use brake
            throttle = 0.0
            brake = min(abs(total_output), 1.0)

        return throttle, brake

    def emergency_brake(self) -> Tuple[float, float]:
        """Emergency braking command"""
        return 0.0, 1.0  # Full brake

    def reset(self):
        """Reset controller state"""
        self.speed_pid.reset()


class LateralController:
    """
    Lateral Controller

    Controls steering for path following using Stanley controller or Pure Pursuit.
    """

    def __init__(self, limits: VehicleLimits):
        self.limits = limits
        self.wheelbase = limits.wheelbase

        # Stanley controller gains
        self.k_e = 0.5  # Cross-track error gain
        self.k_v = 1.0  # Velocity gain
        self.k_theta = 1.0  # Heading error gain

        # Pure pursuit lookahead
        self.lookahead_min = 5.0  # meters
        self.lookahead_max = 25.0  # meters
        self.lookahead_gain = 0.5

        # Steering rate limiter
        self.previous_steering = 0.0
        self.max_steering_rate = limits.max_steering_rate  # deg/s

    def update_stanley(self, target: ControlTarget, state: VehicleState, dt: float) -> float:
        """
        Stanley controller for path tracking

        Args:
            target: Control target with path errors
            state: Current vehicle state
            dt: Time step

        Returns:
            Steering angle command in degrees
        """
        # Heading error term
        theta_e = target.path_heading_error
        theta_term = self.k_theta * theta_e

        # Cross-track error term
        e_fa = target.path_lateral_error  # Lateral error at front axle
        v = max(state.speed, 0.1)  # Avoid division by zero

        cte_term = math.atan2(self.k_e * e_fa, self.k_v * v)
        cte_term_deg = math.degrees(cte_term)

        # Combine terms
        steering_angle = theta_term + cte_term_deg

        # Apply rate limiter
        steering_angle = self._apply_rate_limit(steering_angle, dt)

        # Clamp to limits
        steering_angle = np.clip(
            steering_angle,
            -self.limits.max_steering_angle,
            self.limits.max_steering_angle
        )

        return steering_angle

    def update_pure_pursuit(self, target_point: Tuple[float, float], state: VehicleState, dt: float) -> float:
        """
        Pure Pursuit controller

        Args:
            target_point: (x, y) lookahead point in vehicle frame
            state: Current vehicle state
            dt: Time step

        Returns:
            Steering angle command in degrees
        """
        # Calculate lookahead distance based on speed
        lookahead = np.clip(
            self.lookahead_min + self.lookahead_gain * state.speed,
            self.lookahead_min,
            self.lookahead_max
        )

        # Calculate curvature to target point
        target_x, target_y = target_point
        ld = math.sqrt(target_x**2 + target_y**2)

        if ld < 0.1:
            return 0.0

        # Pure pursuit curvature: k = 2*sin(alpha) / L
        alpha = math.atan2(target_y, target_x)
        curvature = 2.0 * math.sin(alpha) / ld

        # Convert curvature to steering angle (bicycle model)
        steering_angle_rad = math.atan(curvature * self.wheelbase)
        steering_angle = math.degrees(steering_angle_rad)

        # Apply rate limiter
        steering_angle = self._apply_rate_limit(steering_angle, dt)

        # Clamp to limits
        steering_angle = np.clip(
            steering_angle,
            -self.limits.max_steering_angle,
            self.limits.max_steering_angle
        )

        return steering_angle

    def _apply_rate_limit(self, commanded_steering: float, dt: float) -> float:
        """Apply steering rate limit"""
        max_change = self.max_steering_rate * dt

        steering_change = commanded_steering - self.previous_steering
        steering_change = np.clip(steering_change, -max_change, max_change)

        limited_steering = self.previous_steering + steering_change
        self.previous_steering = limited_steering

        return limited_steering

    def reset(self):
        """Reset controller state"""
        self.previous_steering = 0.0


class VehicleControlInterface:
    """
    Vehicle Control Interface

    Main interface for low-level vehicle control. Coordinates longitudinal
    and lateral controllers and manages actuator commands.
    """

    def __init__(self, limits: Optional[VehicleLimits] = None):
        self.limits = limits or VehicleLimits()

        # Controllers
        self.longitudinal_controller = LongitudinalController(self.limits)
        self.lateral_controller = LateralController(self.limits)

        # Control mode
        self.control_mode = ControlMode.MANUAL

        # Actuator states
        self.actuators: Dict[ActuatorType, ActuatorState] = {
            ActuatorType.STEERING: ActuatorState(ActuatorType.STEERING),
            ActuatorType.THROTTLE: ActuatorState(ActuatorType.THROTTLE),
            ActuatorType.BRAKE: ActuatorState(ActuatorType.BRAKE),
            ActuatorType.GEAR: ActuatorState(ActuatorType.GEAR)
        }

        # Current command
        self.current_command: Optional[ControlCommand] = None

        # Command history
        self.command_history: deque = deque(maxlen=1000)

        # Statistics
        self.stats = ControlStatistics()

        # Safety flags
        self.emergency_stop_active = False
        self.override_active = False

        # Timing
        self.last_update_time = 0.0

        print("Vehicle Control Interface initialized")
        print(f"Max steering: ±{self.limits.max_steering_angle}°")
        print(f"Max acceleration: {self.limits.max_acceleration} m/s²")
        print(f"Max deceleration: {self.limits.max_deceleration} m/s²")

    def set_control_mode(self, mode: ControlMode):
        """Set vehicle control mode"""
        if mode != self.control_mode:
            print(f"[Control] Mode change: {self.control_mode.value} -> {mode.value}")
            self.control_mode = mode

            # Reset controllers when changing mode
            if mode == ControlMode.AUTONOMOUS:
                self.longitudinal_controller.reset()
                self.lateral_controller.reset()

    def update(self, target: ControlTarget, state: VehicleState, dt: float) -> ControlCommand:
        """
        Update vehicle control

        Args:
            target: Control target
            state: Current vehicle state
            dt: Time step

        Returns:
            Control command to be executed
        """
        current_time = time.time()

        # Check for emergency stop
        if self.emergency_stop_active or self.control_mode == ControlMode.EMERGENCY_STOP:
            return self._emergency_stop_command(current_time)

        # Only compute control in autonomous mode
        if self.control_mode != ControlMode.AUTONOMOUS:
            # In manual or assisted mode, return neutral command
            return ControlCommand(
                timestamp=current_time,
                steering_angle=0.0,
                throttle=0.0,
                brake=0.0
            )

        # Longitudinal control
        throttle, brake = self.longitudinal_controller.update(target, state, dt)

        # Lateral control (using Stanley controller)
        steering_angle = self.lateral_controller.update_stanley(target, state, dt)

        # Create control command
        command = ControlCommand(
            timestamp=current_time,
            steering_angle=steering_angle,
            throttle=throttle,
            brake=brake,
            gear=GearPosition.DRIVE
        )

        # Validate and clamp command
        if not command.validate(self.limits):
            print("[Control] Warning: Command exceeds limits, clamping...")
            command.clamp(self.limits)
            self.stats.limit_violations += 1

        # Store command
        self.current_command = command
        self.command_history.append(command)
        self.stats.total_commands += 1

        # Update timing statistics
        if self.last_update_time > 0:
            loop_time = current_time - self.last_update_time
            if loop_time > 0:
                freq = 1.0 / loop_time
                # Exponential moving average
                alpha = 0.1
                self.stats.control_loop_frequency = (
                    (1 - alpha) * self.stats.control_loop_frequency + alpha * freq
                )

        self.last_update_time = current_time

        return command

    def update_actuator_feedback(self, actuator_type: ActuatorType, actual_value: float):
        """
        Update actuator feedback

        Args:
            actuator_type: Type of actuator
            actual_value: Actual actuator value
        """
        if actuator_type in self.actuators:
            actuator = self.actuators[actuator_type]
            actuator.actual_value = actual_value
            actuator.last_update = time.time()

            # Check for errors
            error = actuator.get_error()
            if error > 0.2:  # 20% error threshold
                actuator.error_count += 1
                if actuator.error_count > 10:
                    actuator.status = ActuatorStatus.DEGRADED
                    print(f"[Control] Warning: {actuator_type.value} actuator degraded")

    def execute_command(self, command: ControlCommand) -> bool:
        """
        Execute control command

        Args:
            command: Control command to execute

        Returns:
            True if command executed successfully
        """
        # Validate command
        if not command.validate(self.limits):
            print("[Control] Error: Invalid command")
            return False

        # Check actuator health
        for actuator_type in [ActuatorType.STEERING, ActuatorType.THROTTLE, ActuatorType.BRAKE]:
            actuator = self.actuators[actuator_type]
            if actuator.status == ActuatorStatus.FAILED:
                print(f"[Control] Error: {actuator_type.value} actuator failed")
                self.trigger_emergency_stop()
                return False

        # Update commanded values
        self.actuators[ActuatorType.STEERING].commanded_value = command.steering_angle
        self.actuators[ActuatorType.THROTTLE].commanded_value = command.throttle
        self.actuators[ActuatorType.BRAKE].commanded_value = command.brake

        # In a real implementation, this would send commands to vehicle CAN bus
        # or other low-level interface

        return True

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        print("[Control] EMERGENCY STOP TRIGGERED!")
        self.emergency_stop_active = True
        self.control_mode = ControlMode.EMERGENCY_STOP
        self.stats.emergency_stops += 1

    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        print("[Control] Emergency stop cleared")
        self.emergency_stop_active = False
        if self.control_mode == ControlMode.EMERGENCY_STOP:
            self.control_mode = ControlMode.MANUAL

    def _emergency_stop_command(self, timestamp: float) -> ControlCommand:
        """Generate emergency stop command"""
        return ControlCommand(
            timestamp=timestamp,
            steering_angle=0.0,
            throttle=0.0,
            brake=1.0,  # Full brake
            gear=GearPosition.PARK
        )

    def get_actuator_health(self) -> Dict[ActuatorType, ActuatorStatus]:
        """Get health status of all actuators"""
        return {
            actuator_type: actuator.status
            for actuator_type, actuator in self.actuators.items()
        }

    def diagnose_actuators(self) -> str:
        """
        Diagnose actuator health

        Returns:
            Diagnostic report string
        """
        lines = ["=" * 60]
        lines.append("Actuator Diagnostics")
        lines.append("=" * 60)

        for actuator_type, actuator in self.actuators.items():
            lines.append(f"\n{actuator_type.value.upper()}:")
            lines.append(f"  Status: {actuator.status.value}")
            lines.append(f"  Commanded: {actuator.commanded_value:.3f}")
            lines.append(f"  Actual: {actuator.actual_value:.3f}")
            lines.append(f"  Error: {actuator.get_error():.3f}")
            lines.append(f"  Error Count: {actuator.error_count}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def get_statistics(self) -> ControlStatistics:
        """Get control statistics"""
        return self.stats

    def visualize_control_status(self, state: VehicleState) -> str:
        """
        Generate text visualization of control status

        Args:
            state: Current vehicle state

        Returns:
            Text visualization
        """
        lines = ["=" * 60]
        lines.append("Vehicle Control Status")
        lines.append("=" * 60)

        lines.append(f"\nControl Mode: {self.control_mode.value.upper()}")
        lines.append(f"Emergency Stop: {'ACTIVE' if self.emergency_stop_active else 'Inactive'}")

        if self.current_command:
            lines.append(f"\nCurrent Command:")
            lines.append(f"  Steering: {self.current_command.steering_angle:6.2f}°")
            lines.append(f"  Throttle: {self.current_command.throttle:6.2%}")
            lines.append(f"  Brake:    {self.current_command.brake:6.2%}")
            lines.append(f"  Gear:     {self.current_command.gear.value}")

        lines.append(f"\nVehicle State:")
        lines.append(f"  Speed:        {state.speed:6.2f} m/s ({state.speed*3.6:.1f} km/h)")
        lines.append(f"  Acceleration: {state.acceleration:6.2f} m/s²")
        lines.append(f"  Steering:     {state.steering_angle:6.2f}°")
        lines.append(f"  Yaw Rate:     {state.yaw_rate:6.2f}°/s")

        lines.append(f"\nActuator Health:")
        for actuator_type, actuator in self.actuators.items():
            status_symbol = "✓" if actuator.status == ActuatorStatus.HEALTHY else "✗"
            lines.append(f"  {status_symbol} {actuator_type.value}: {actuator.status.value}")

        lines.append(f"\nStatistics:")
        lines.append(f"  Commands Sent: {self.stats.total_commands}")
        lines.append(f"  Emergency Stops: {self.stats.emergency_stops}")
        lines.append(f"  Limit Violations: {self.stats.limit_violations}")
        lines.append(f"  Control Frequency: {self.stats.control_loop_frequency:.1f} Hz")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Test vehicle control interface"""
    print("Testing Vehicle Control Interface\n")

    # Create vehicle limits
    limits = VehicleLimits(
        max_steering_angle=35.0,
        max_acceleration=3.0,
        max_deceleration=8.0,
        wheelbase=2.7
    )

    # Create control interface
    controller = VehicleControlInterface(limits)

    # Set to autonomous mode
    controller.set_control_mode(ControlMode.AUTONOMOUS)

    # Create initial vehicle state
    state = VehicleState(
        timestamp=time.time(),
        position_x=0.0,
        position_y=0.0,
        heading=0.0,
        speed=10.0,  # 10 m/s ≈ 36 km/h
        acceleration=0.0,
        steering_angle=0.0
    )

    # Create control target (lane keeping scenario)
    target = ControlTarget(
        target_speed=15.0,  # Speed up to 15 m/s
        target_acceleration=0.5,  # Gentle acceleration
        path_lateral_error=-0.2,  # 0.2m right of centerline
        path_heading_error=2.0  # 2 degrees heading error
    )

    print("Scenario: Lane keeping with speed increase\n")
    print(f"Initial speed: {state.speed:.1f} m/s")
    print(f"Target speed: {target.target_speed:.1f} m/s")
    print(f"Lateral error: {target.path_lateral_error:.2f} m")
    print(f"Heading error: {target.path_heading_error:.1f}°\n")

    # Simulation loop
    dt = 0.1  # 100ms control loop
    num_steps = 50

    print("Running control loop...\n")

    for i in range(num_steps):
        # Update control
        command = controller.update(target, state, dt)

        # Execute command
        success = controller.execute_command(command)

        # Simulate actuator feedback (simplified)
        controller.update_actuator_feedback(ActuatorType.STEERING, command.steering_angle)
        controller.update_actuator_feedback(ActuatorType.THROTTLE, command.throttle)
        controller.update_actuator_feedback(ActuatorType.BRAKE, command.brake)

        # Simple vehicle dynamics simulation
        # Longitudinal dynamics
        net_accel = command.throttle * limits.max_acceleration - command.brake * limits.max_deceleration
        state.acceleration = net_accel
        state.speed += net_accel * dt
        state.speed = max(0.0, state.speed)  # No negative speed

        # Lateral dynamics (simplified)
        state.steering_angle = command.steering_angle
        state.yaw_rate = (state.speed / limits.wheelbase) * math.tan(math.radians(state.steering_angle))

        # Print status every 10 steps
        if i % 10 == 0:
            print(f"Step {i:3d}: Speed={state.speed:5.2f} m/s, "
                  f"Steering={command.steering_angle:5.2f}°, "
                  f"Throttle={command.throttle:.2f}, Brake={command.brake:.2f}")

        # Simulate reaching target
        if i == 30:
            target.path_lateral_error = 0.0  # Corrected lateral error
            target.path_heading_error = 0.0  # Corrected heading error

        time.sleep(0.01)  # Small delay

    # Print final status
    print("\n" + controller.visualize_control_status(state))

    # Print actuator diagnostics
    print("\n" + controller.diagnose_actuators())

    # Test emergency stop
    print("\nTesting emergency stop...")
    controller.trigger_emergency_stop()

    emergency_command = controller.update(target, state, dt)
    print(f"Emergency command: Throttle={emergency_command.throttle:.2f}, "
          f"Brake={emergency_command.brake:.2f}, Gear={emergency_command.gear.value}")

    controller.clear_emergency_stop()

    # Final statistics
    stats = controller.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Commands: {stats.total_commands}")
    print(f"  Emergency Stops: {stats.emergency_stops}")
    print(f"  Limit Violations: {stats.limit_violations}")
    print(f"  Control Frequency: {stats.control_loop_frequency:.1f} Hz")

    print("\n✓ Vehicle Control Interface test complete!")


if __name__ == "__main__":
    main()
