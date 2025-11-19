# Integrated Autonomous Vehicle System - User Guide

## Version 1.3.0 - Advanced Planning & Connected Systems

## Overview

The Integrated Autonomous Vehicle System is a comprehensive GUI application that brings together **all features** from versions 1.0 through 1.3 of the Autonomous Vehicle Perception System. This provides a unified interface for:

- Multi-camera perception and tracking
- Advanced ADAS features
- Complex planning capabilities
- Motion planning and vehicle control
- V2X communication
- Scenario testing and validation

**Total system capabilities**: 16,500+ lines of production-ready code integrated into a single application.

## Quick Start

### Launch the Application

#### Option 1: Using the Launcher Script (Recommended)
```bash
./launch_integrated_system.sh
```

#### Option 2: Direct Python Execution
```bash
python3 integrated_av_system.py
```

### First Time Setup

1. **Install Dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python3 -c "import PyQt6; import cv2; print('Ready!')"
   ```

3. **Launch Application** and you should see the main window with 7 tabs.

## Application Structure

The integrated system is organized into **7 main tabs**, each focusing on a different aspect of autonomous driving:

### 1. 📹 Perception & Cameras

**Purpose**: Core perception capabilities including camera feeds, object detection, and sensor fusion.

**Features**:
- **Camera Grid**: 2x2 grid showing 4 camera feeds (Dashboard, Front, Left, Right)
- **Perception Modules**:
  - ✓ Lane Detection
  - ✓ Object Detection (YOLOv8)
  - ✓ Object Tracking
  - ✓ Traffic Light Detection
  - ✓ Depth Estimation
  - ✓ Semantic Segmentation
  - ✓ Scene Recognition

- **Sensor Fusion**:
  - Multi-Camera Fusion (v1.0)
  - Advanced EKF Fusion (v1.3)

- **Visualization Options**:
  - Bird's Eye View (BEV)
  - Trajectory visualization
  - Camera overlays

**How to Use**:
1. Enable desired perception modules using checkboxes
2. Select fusion method
3. Enable visualization options
4. Start the system using the master START button

### 2. 🛡️ Advanced ADAS

**Purpose**: Advanced Driver Assistance Systems for safety and awareness.

**Features**:

#### Traffic Sign Recognition (TSR)
- Detects 40+ traffic sign types
- Speed limits, regulatory signs, warning signs, informational signs
- Multi-frame tracking for stability
- Distance estimation

#### Driver Monitoring System (DMS)
- Face and eye detection
- Drowsiness detection
- Distraction detection
- Attention scoring (0-100)
- Real-time alerts

#### 3D Object Detection
- Full 3D bounding boxes
- 6-DOF pose estimation
- Depth integration

#### Predictive Collision Warning (PCW)
- 5-second ahead prediction
- 9 collision scenario types
- Risk level assessment
- Recommended actions

#### Real-time Safety Scoring
- Overall safety score (0-100)
- Component scores:
  - Collision Avoidance (35%)
  - Lane Keeping (20%)
  - Following Distance (20%)
  - Speed Appropriateness (15%)
  - Environmental Awareness (10%)

**How to Use**:
1. Enable desired ADAS features
2. Monitor attention scores and safety metrics
3. Review detected signs and warnings
4. Observe real-time safety scoring

### 3. 🗺️ Planning & Navigation

**Purpose**: Path planning, route planning, and parking assistance.

**Features**:

#### Path Planning
- **Algorithms**: Polynomial, Quintic, A*, RRT, Frenet
- **Maneuvers**: Lane Keeping, Lane Change, Overtake, Emergency Stop
- Obstacle avoidance
- Speed profile optimization

#### Motion Planning with Vehicle Dynamics
- **Vehicle Models**:
  - Kinematic Bicycle
  - Dynamic Bicycle
  - Point Mass
- Lattice-based trajectory planning
- Constraint checking

#### Global Route Planning
- Lane graph construction
- A* pathfinding
- Alternative route generation
- Turn-by-turn instructions

#### Parking Assist
- **Modes**: Parallel, Perpendicular, Angled (45°, 60°)
- Automatic space detection
- Multi-point turn planning
- Collision-free trajectories

**How to Use**:
1. Select path planning algorithm
2. Choose maneuver type
3. Enable motion planning with desired vehicle model
4. Set up route planning
5. Configure parking mode if needed
6. View planning visualization

### 4. 🎮 Motion Control

**Purpose**: Low-level vehicle control with PID and advanced controllers.

**Features**:

#### Control Modes
- Manual
- Assisted
- Autonomous
- Emergency Stop

#### Longitudinal Control
- Target speed setting
- Target acceleration
- PID controller
- Throttle and brake coordination

#### Lateral Control
- **Controllers**:
  - Stanley Controller
  - Pure Pursuit Controller
- Lateral error compensation
- Steering rate limiting

#### Control Monitoring
- Current command display (steering, throttle, brake)
- Vehicle state (speed, acceleration, yaw rate)
- Actuator health diagnostics
- Control statistics

**How to Use**:
1. Select control mode
2. Set target speed and acceleration
3. Choose lateral controller
4. Monitor control commands and vehicle state
5. Check actuator health status

### 5. 📡 V2X Communication

**Purpose**: Vehicle-to-Everything communication for connected driving.

**Features**:

#### Protocols
- DSRC (Dedicated Short Range Communications)
- C-V2X (Cellular V2X)
- Hybrid mode

#### Message Types
- **BSM**: Basic Safety Message (vehicle state broadcast)
- **SPaT**: Signal Phase and Timing (traffic signals)
- **MAP**: Map Data (road geometry)
- **PSM**: Personal Safety Message (pedestrians)
- **RSA**: Road Side Alert (hazards)

#### Cooperative Awareness
- Remote vehicle tracking
- Infrastructure node communication
- Road alert monitoring
- Time-to-collision via V2X

#### V2X Statistics
- Messages sent/received
- Message drop rate
- Average latency

**How to Use**:
1. Enable V2X communication
2. Select protocol (DSRC/C-V2X)
3. Set BSM broadcast frequency
4. Enable desired message types
5. Enable cooperative awareness
6. Monitor remote vehicles and infrastructure

### 6. 🧪 Testing & Validation

**Purpose**: Automated scenario testing for system validation.

**Features**:

#### Predefined Scenarios
- Highway Cruise
- Emergency Braking
- Lane Change
- Cut-In
- Urban Intersection
- Parking Maneuver
- Adverse Weather
- Custom Scenario

#### Test Execution
- Single scenario testing
- Full test suite execution
- Progress monitoring
- Performance metrics collection

#### Test Results
- Pass/fail criteria evaluation
- Safety metrics (TTC violations, collisions, min distances)
- Comfort metrics (acceleration, jerk, lateral acceleration)
- Text and JSON reports

**How to Use**:
1. Select test scenario from dropdown
2. Click "Run Test" for single scenario
3. Click "Run Test Suite" for all scenarios
4. Monitor test progress
5. Review test results and metrics

### 7. 📊 System Monitor

**Purpose**: Real-time system monitoring and diagnostics.

**Features**:

#### System Statistics
- Frames processed
- Objects detected
- Paths planned
- Controls sent
- V2X messages
- Average FPS
- CPU usage
- Memory usage

#### Module Status
- Perception System status
- Planning System status
- Control System status
- V2X Communication status
- Safety Monitor status
- Recording System status

#### System Log
- Real-time event logging
- Info, warning, and error messages
- System state changes

**How to Use**:
1. Monitor system statistics in real-time
2. Check module status indicators
3. Review system log for events
4. Track resource usage

## Master Controls

### Footer Controls (Always Visible)

#### 🚀 START SYSTEM Button
- **Green**: System is stopped, click to start
- **Yellow**: System is running, click to stop
- Starts/stops all enabled modules

#### ⛔ EMERGENCY STOP Button
- **Red**: Immediate system halt
- All systems stopped instantly
- Use in critical situations

#### System Status Display
- Shows current system state:
  - STOPPED (Yellow)
  - RUNNING (Green)
  - EMERGENCY STOP (Red)

#### Uptime Display
- Shows session uptime in HH:MM:SS format
- Resets on each START

## Menu Bar

### File Menu
- **Load Configuration**: Load saved system configuration
- **Save Configuration**: Save current configuration
- **Exit**: Close application

### View Menu
- **Toggle Fullscreen**: Switch between windowed and fullscreen mode

### Tools Menu
- **Camera Calibration**: Launch camera calibration wizard

### Help Menu
- **About**: Display application information

## Workflow Examples

### Example 1: Basic Perception Session

1. Launch application
2. Go to **Perception & Cameras** tab
3. Enable:
   - Lane Detection ✓
   - Object Detection ✓
   - Object Tracking ✓
   - Multi-Camera Fusion ✓
4. Enable Bird's Eye View visualization
5. Click **START SYSTEM**
6. Monitor camera feeds and BEV

### Example 2: Full Autonomous Driving Session

1. Launch application
2. **Perception Tab**: Enable all perception modules
3. **ADAS Tab**: Enable TSR, DMS, PCW, Safety Scoring
4. **Planning Tab**:
   - Enable Path Planning (Frenet algorithm)
   - Enable Motion Planning (Kinematic Bicycle)
   - Enable Route Planning
5. **Control Tab**:
   - Set to Autonomous mode
   - Set target speed: 15 m/s
   - Select Stanley controller
6. **V2X Tab**: Enable V2X with DSRC protocol
7. Click **START SYSTEM**
8. Monitor all tabs for system operation
9. Go to **System Monitor** to check performance

### Example 3: Testing and Validation

1. Launch application
2. Go to **Testing & Validation** tab
3. Select "Highway Cruise" scenario
4. Click **Run Test**
5. Monitor test progress bar
6. Review test results
7. Click **Run Test Suite** to test all scenarios
8. Export results for analysis

### Example 4: Parking Assistance

1. **Planning Tab**:
   - Enable Parking Assist
   - Select parking mode: Parallel
2. **Control Tab**:
   - Set to Assisted mode
3. **Perception Tab**:
   - Enable Object Detection
   - Enable Depth Estimation
4. Click **START SYSTEM**
5. System will detect parking spaces and assist with parking

## Configuration Tips

### Performance Optimization

1. **For Low-End Hardware**:
   - Disable heavy modules (Depth Estimation, Semantic Segmentation)
   - Use simpler path planning algorithms
   - Reduce camera resolution

2. **For High-End Hardware**:
   - Enable all perception modules
   - Use advanced algorithms (RRT, Frenet)
   - Enable Advanced EKF Fusion
   - High-resolution cameras

### Safety-Critical Applications

1. **Enable All Safety Features**:
   - Predictive Collision Warning
   - Safety Scoring
   - Driver Monitoring
   - V2X Communication

2. **Run Validation Tests**:
   - Before deployment, run full test suite
   - Ensure all critical scenarios pass
   - Review safety metrics

### Development and Testing

1. **Use Scenario Testing**:
   - Test individual features with specific scenarios
   - Validate performance metrics
   - Generate reports for analysis

2. **Monitor System Performance**:
   - Keep System Monitor tab open
   - Watch for resource usage
   - Check module status indicators

## Keyboard Shortcuts

- **F11**: Toggle fullscreen
- **Ctrl+Q**: Quit application
- **Space**: Start/Stop system (when in focus)

## Troubleshooting

### Application Won't Start

1. Check Python version: `python3 --version` (need 3.10+)
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
3. Check logs in `data/logs/`

### Camera Feeds Not Showing

1. Ensure cameras are connected
2. Check camera permissions
3. Go to original `main.py` to test cameras individually
4. Run camera discovery tool

### Low Performance

1. Check System Monitor tab for resource usage
2. Disable unused modules
3. Reduce camera resolution
4. Close other applications

### Modules Not Responding

1. Check module status in System Monitor
2. Review System Log for errors
3. Try EMERGENCY STOP and restart
4. Check individual module files for issues

## Advanced Features

### Custom Scenarios

To create custom test scenarios:
1. Go to Testing & Validation tab
2. Select "Custom Scenario"
3. Define:
   - Scenario parameters
   - Initial conditions
   - Pass/fail criteria
4. Run and evaluate

### Integration with External Systems

The integrated system can be extended to interface with:
- Real vehicle CAN bus
- External sensors (LIDAR, Radar)
- Simulation environments (CARLA, SUMO)
- Cloud services for data logging

See integration guides in `docs/` for details.

## System Architecture

### Module Communication

```
┌─────────────────────────────────────────────┐
│         Integrated AV System GUI            │
├──────────┬─────────┬─────────┬─────────────┤
│Perception│ Planning│ Control │     V2X     │
├──────────┴─────────┴─────────┴─────────────┤
│          Sensor Fusion Framework           │
├────────────────────────────────────────────┤
│    Camera Manager  │  Data Structures      │
└────────────────────────────────────────────┘
```

### Data Flow

1. **Perception** → Sensor Fusion
2. **Sensor Fusion** → Planning
3. **Planning** → Control
4. **Control** → Vehicle/Simulation
5. **V2X** ↔ All modules (cooperative awareness)

## Best Practices

1. **Always start with Perception**:
   - Enable basic perception modules first
   - Verify camera feeds
   - Then enable advanced features

2. **Monitor Safety Scoring**:
   - Keep safety score above 70
   - Address any warnings promptly

3. **Use Scenario Testing**:
   - Before deploying new features
   - After configuration changes
   - Regularly for validation

4. **Enable V2X in Multi-Vehicle Scenarios**:
   - For cooperative awareness
   - Enhanced safety
   - Better decision making

5. **Regular System Monitoring**:
   - Check System Monitor tab periodically
   - Watch for resource spikes
   - Review system logs

## Support and Resources

- **User Manual**: `docs/USER_MANUAL.md`
- **API Documentation**: Individual module files have comprehensive docstrings
- **Phase Guides**: `docs/` directory contains integration guides for each phase
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Examples**: `examples/` directory contains usage examples

## Version History

- **v1.3.0**: Integrated all features into single GUI
  - Motion planning with vehicle dynamics
  - Advanced sensor fusion (EKF)
  - Route planning (A*)
  - V2X communication
  - Vehicle control interface
  - Scenario testing framework

- **v1.2.0**: Complex planning features
  - Path planning
  - Traffic sign recognition
  - Driver monitoring
  - 3D object detection
  - Parking assist
  - Predictive collision warning
  - Visual odometry

- **v1.1.0**: Advanced ADAS features
  - Multi-camera DVR
  - Traffic light detection
  - Depth estimation
  - Semantic segmentation
  - Scene recognition
  - Safety scoring
  - Trip analytics

- **v1.0.x**: Core platform
  - Multi-camera capture
  - Lane detection
  - Object detection and tracking
  - Sensor fusion
  - Bird's eye view
  - Safety warnings

## License

See main README.md for license information.

## Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

---

**Built with ❤️ for autonomous vehicle development**
