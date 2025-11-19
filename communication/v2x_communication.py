"""
V2X Communication Module
Implements Vehicle-to-Everything (V2X) communication for connected autonomous vehicles.

Features:
- V2V (Vehicle-to-Vehicle) communication
- V2I (Vehicle-to-Infrastructure) communication
- V2P (Vehicle-to-Pedestrian) communication
- Message types: BSM, SPaT, MAP, PSM
- DSRC and C-V2X protocol support
- Security and authentication
- Message prioritization and queuing
- Cooperative awareness

Author: Autonomous Driving System
Version: 1.3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import time
import hashlib
import json
from collections import deque
import math


class V2XProtocol(Enum):
    """V2X communication protocol types"""
    DSRC = "dsrc"  # Dedicated Short Range Communications
    C_V2X = "c_v2x"  # Cellular V2X
    HYBRID = "hybrid"


class MessageType(Enum):
    """V2X message types"""
    BSM = "basic_safety_message"  # Vehicle state broadcast
    SPaT = "signal_phase_timing"  # Traffic signal information
    MAP = "map_data"  # Road geometry information
    PSM = "personal_safety_message"  # Pedestrian/cyclist information
    RSA = "road_side_alert"  # Road hazard warnings
    TIM = "traveler_information"  # General information
    SSM = "signal_status_message"  # Signal request status
    EVA = "emergency_vehicle_alert"  # Emergency vehicle notification


class MessagePriority(Enum):
    """Message transmission priority"""
    EMERGENCY = 0  # Highest priority
    SAFETY_CRITICAL = 1
    SAFETY = 2
    TRAFFIC_MANAGEMENT = 3
    INFORMATION = 4  # Lowest priority


class SignalPhase(Enum):
    """Traffic signal phase states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    FLASHING_RED = "flashing_red"
    FLASHING_YELLOW = "flashing_yellow"
    UNKNOWN = "unknown"


@dataclass
class Position3D:
    """3D position with latitude, longitude, altitude"""
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float = 0.0  # meters

    def distance_to(self, other: 'Position3D') -> float:
        """Calculate distance using Haversine formula"""
        R = 6371000  # Earth radius in meters

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        distance_2d = R * c
        dalt = other.altitude - self.altitude

        return math.sqrt(distance_2d**2 + dalt**2)


@dataclass
class VehicleState:
    """Complete vehicle state for V2X transmission"""
    vehicle_id: str
    timestamp: float
    position: Position3D
    heading: float  # degrees, 0=North
    speed: float  # m/s
    acceleration: float  # m/s^2
    steering_angle: float = 0.0  # degrees
    yaw_rate: float = 0.0  # deg/s
    vehicle_length: float = 4.5  # meters
    vehicle_width: float = 2.0  # meters
    vehicle_type: str = "passenger_car"


@dataclass
class BasicSafetyMessage:
    """BSM - Basic Safety Message (SAE J2735)"""
    msg_type: MessageType = MessageType.BSM
    msg_id: str = ""
    vehicle_state: Optional[VehicleState] = None
    transmission_time: float = 0.0
    priority: MessagePriority = MessagePriority.SAFETY

    def __post_init__(self):
        if not self.msg_id and self.vehicle_state:
            self.msg_id = f"BSM_{self.vehicle_state.vehicle_id}_{int(time.time()*1000)}"
        if self.transmission_time == 0.0:
            self.transmission_time = time.time()


@dataclass
class SignalPhaseAndTiming:
    """SPaT - Signal Phase and Timing Message"""
    msg_type: MessageType = MessageType.SPaT
    msg_id: str = ""
    intersection_id: str = ""
    timestamp: float = 0.0
    signal_groups: Dict[int, Tuple[SignalPhase, float]] = field(default_factory=dict)  # group_id -> (phase, time_remaining)
    priority: MessagePriority = MessagePriority.TRAFFIC_MANAGEMENT

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"SPaT_{self.intersection_id}_{int(time.time()*1000)}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class MapData:
    """MAP - Intersection/Road geometry"""
    msg_type: MessageType = MessageType.MAP
    msg_id: str = ""
    intersection_id: str = ""
    reference_position: Optional[Position3D] = None
    lanes: List[Dict] = field(default_factory=list)  # Lane geometry data
    priority: MessagePriority = MessagePriority.INFORMATION

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"MAP_{self.intersection_id}_{int(time.time()*1000)}"


@dataclass
class PersonalSafetyMessage:
    """PSM - Pedestrian/Cyclist Safety Message"""
    msg_type: MessageType = MessageType.PSM
    msg_id: str = ""
    user_id: str = ""
    timestamp: float = 0.0
    position: Optional[Position3D] = None
    heading: float = 0.0  # degrees
    speed: float = 0.0  # m/s
    user_type: str = "pedestrian"  # pedestrian, cyclist, wheelchair, etc.
    priority: MessagePriority = MessagePriority.SAFETY_CRITICAL

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"PSM_{self.user_id}_{int(time.time()*1000)}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RoadSideAlert:
    """RSA - Road Side Alert (hazards, construction, etc.)"""
    msg_type: MessageType = MessageType.RSA
    msg_id: str = ""
    alert_id: str = ""
    timestamp: float = 0.0
    position: Optional[Position3D] = None
    alert_type: str = "hazard"  # hazard, construction, accident, etc.
    description: str = ""
    severity: int = 1  # 1-5, 5 being most severe
    radius: float = 100.0  # Alert radius in meters
    priority: MessagePriority = MessagePriority.SAFETY

    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = f"RSA_{self.alert_id}_{int(time.time()*1000)}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class V2XMessage:
    """Generic V2X message wrapper"""
    payload: object  # BSM, SPaT, MAP, PSM, RSA, etc.
    protocol: V2XProtocol
    transmission_power: float = 20.0  # dBm
    transmission_range: float = 300.0  # meters
    signature: str = ""  # Security signature
    sequence_number: int = 0


@dataclass
class V2XConfig:
    """V2X communication configuration"""
    protocol: V2XProtocol = V2XProtocol.DSRC
    frequency: float = 5.9  # GHz (5.9 GHz for DSRC)
    bandwidth: float = 10.0  # MHz
    transmission_power: float = 20.0  # dBm
    max_range: float = 300.0  # meters
    bsm_frequency: float = 10.0  # Hz (transmit BSM at 10 Hz)
    message_queue_size: int = 1000
    enable_security: bool = True
    enable_cooperative_awareness: bool = True
    min_distance_threshold: float = 150.0  # meters

    # Protocol-specific settings
    dsrc_channel: int = 172  # DSRC control channel
    cv2x_bandwidth: str = "20MHz"


@dataclass
class RemoteVehicle:
    """Tracking information for remote vehicles"""
    vehicle_id: str
    last_state: VehicleState
    last_update: float
    message_count: int = 0
    lost_message_count: int = 0
    average_rssi: float = -80.0  # dBm

    def is_stale(self, current_time: float, timeout: float = 1.0) -> bool:
        """Check if vehicle data is stale"""
        return (current_time - self.last_update) > timeout


@dataclass
class V2XStatistics:
    """V2X communication statistics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    average_latency: float = 0.0  # seconds
    average_packet_loss: float = 0.0  # percentage
    remote_vehicles_count: int = 0
    infrastructure_nodes_count: int = 0
    pedestrians_count: int = 0


class V2XCommunication:
    """
    V2X Communication System

    Handles all Vehicle-to-Everything communication including V2V, V2I, V2P.
    Implements message encoding/decoding, security, and cooperative awareness.
    """

    def __init__(self, vehicle_id: str, config: Optional[V2XConfig] = None):
        self.vehicle_id = vehicle_id
        self.config = config or V2XConfig()

        # Message queues (priority queues)
        self.outgoing_queue: List[Tuple[float, V2XMessage]] = []  # (priority, message)
        self.incoming_queue: deque = deque(maxlen=self.config.message_queue_size)

        # Remote entity tracking
        self.remote_vehicles: Dict[str, RemoteVehicle] = {}
        self.infrastructure_nodes: Dict[str, Dict] = {}
        self.pedestrians: Dict[str, PersonalSafetyMessage] = {}

        # Security
        self.private_key = self._generate_private_key()
        self.certificate = self._generate_certificate()

        # Statistics
        self.stats = V2XStatistics()

        # Internal state
        self.current_vehicle_state: Optional[VehicleState] = None
        self.last_bsm_time = 0.0
        self.sequence_number = 0

        print(f"V2X Communication initialized for vehicle {vehicle_id}")
        print(f"Protocol: {self.config.protocol.value}")
        print(f"Max range: {self.config.max_range}m, BSM rate: {self.config.bsm_frequency}Hz")

    def update_vehicle_state(self, state: VehicleState):
        """Update current vehicle state"""
        self.current_vehicle_state = state

    def update(self, dt: float):
        """
        Update V2X communication system

        Args:
            dt: Time step in seconds
        """
        current_time = time.time()

        # Periodic BSM transmission
        if self.current_vehicle_state and self._should_send_bsm(current_time):
            self._send_bsm()

        # Process incoming messages
        self._process_incoming_messages(current_time)

        # Clean up stale remote vehicles
        self._cleanup_stale_vehicles(current_time)

        # Update cooperative awareness
        if self.config.enable_cooperative_awareness:
            self._update_cooperative_awareness(current_time)

        # Transmit queued messages
        self._transmit_messages()

    def _should_send_bsm(self, current_time: float) -> bool:
        """Determine if BSM should be sent"""
        time_since_last = current_time - self.last_bsm_time
        return time_since_last >= (1.0 / self.config.bsm_frequency)

    def _send_bsm(self):
        """Create and queue Basic Safety Message"""
        if not self.current_vehicle_state:
            return

        bsm = BasicSafetyMessage(
            vehicle_state=self.current_vehicle_state,
            transmission_time=time.time()
        )

        self.send_message(bsm, MessagePriority.SAFETY)
        self.last_bsm_time = time.time()

    def send_message(self, payload: object, priority: MessagePriority = MessagePriority.INFORMATION):
        """
        Queue a V2X message for transmission

        Args:
            payload: Message payload (BSM, SPaT, MAP, etc.)
            priority: Message priority
        """
        # Create V2X message wrapper
        message = V2XMessage(
            payload=payload,
            protocol=self.config.protocol,
            transmission_power=self.config.transmission_power,
            transmission_range=self.config.max_range,
            sequence_number=self.sequence_number
        )

        self.sequence_number += 1

        # Add security signature
        if self.config.enable_security:
            message.signature = self._sign_message(message)

        # Add to priority queue (lower priority value = higher priority)
        import heapq
        heapq.heappush(self.outgoing_queue, (priority.value, message))

    def _transmit_messages(self):
        """Transmit queued messages"""
        import heapq

        while self.outgoing_queue:
            priority, message = heapq.heappop(self.outgoing_queue)

            # Simulate transmission
            success = self._simulate_transmission(message)

            if success:
                self.stats.messages_sent += 1
                message_bytes = len(json.dumps(self._serialize_message(message)))
                self.stats.bytes_sent += message_bytes
            else:
                self.stats.messages_dropped += 1

    def _simulate_transmission(self, message: V2XMessage) -> bool:
        """
        Simulate wireless transmission with path loss and interference

        Args:
            message: V2X message to transmit

        Returns:
            True if transmission successful
        """
        # Simple path loss model: PL(d) = PL(d0) + 10*n*log10(d/d0)
        # For V2X: n ≈ 2.5-3.5 depending on environment

        # Assume success for now (in real implementation, this would model
        # wireless channel, collisions, interference, etc.)
        transmission_success_rate = 0.95  # 95% success rate

        return np.random.random() < transmission_success_rate

    def receive_message(self, message: V2XMessage, rssi: float = -80.0):
        """
        Receive a V2X message from another entity

        Args:
            message: Received V2X message
            rssi: Received signal strength indicator (dBm)
        """
        # Verify security signature
        if self.config.enable_security:
            if not self._verify_signature(message):
                print(f"[V2X] Message signature verification failed: {message.sequence_number}")
                return

        # Add to incoming queue
        self.incoming_queue.append((message, rssi, time.time()))
        self.stats.messages_received += 1

    def _process_incoming_messages(self, current_time: float):
        """Process messages in incoming queue"""
        while self.incoming_queue:
            message, rssi, receive_time = self.incoming_queue.popleft()

            # Calculate latency
            if hasattr(message.payload, 'transmission_time'):
                latency = receive_time - message.payload.transmission_time
                # Update average latency (exponential moving average)
                alpha = 0.1
                self.stats.average_latency = (1 - alpha) * self.stats.average_latency + alpha * latency

            # Process based on message type
            if isinstance(message.payload, BasicSafetyMessage):
                self._process_bsm(message.payload, rssi)
            elif isinstance(message.payload, SignalPhaseAndTiming):
                self._process_spat(message.payload)
            elif isinstance(message.payload, MapData):
                self._process_map(message.payload)
            elif isinstance(message.payload, PersonalSafetyMessage):
                self._process_psm(message.payload)
            elif isinstance(message.payload, RoadSideAlert):
                self._process_rsa(message.payload)

    def _process_bsm(self, bsm: BasicSafetyMessage, rssi: float):
        """Process received Basic Safety Message"""
        if not bsm.vehicle_state:
            return

        vehicle_id = bsm.vehicle_state.vehicle_id

        if vehicle_id == self.vehicle_id:
            return  # Ignore own messages

        # Update or create remote vehicle entry
        if vehicle_id in self.remote_vehicles:
            remote = self.remote_vehicles[vehicle_id]
            remote.last_state = bsm.vehicle_state
            remote.last_update = time.time()
            remote.message_count += 1
            # Update average RSSI
            alpha = 0.2
            remote.average_rssi = (1 - alpha) * remote.average_rssi + alpha * rssi
        else:
            self.remote_vehicles[vehicle_id] = RemoteVehicle(
                vehicle_id=vehicle_id,
                last_state=bsm.vehicle_state,
                last_update=time.time(),
                message_count=1,
                average_rssi=rssi
            )

    def _process_spat(self, spat: SignalPhaseAndTiming):
        """Process Signal Phase and Timing message"""
        # Store infrastructure information
        self.infrastructure_nodes[spat.intersection_id] = {
            'type': 'traffic_signal',
            'last_update': time.time(),
            'data': spat
        }

    def _process_map(self, map_data: MapData):
        """Process MAP data message"""
        self.infrastructure_nodes[map_data.intersection_id] = {
            'type': 'map_data',
            'last_update': time.time(),
            'data': map_data
        }

    def _process_psm(self, psm: PersonalSafetyMessage):
        """Process Personal Safety Message (pedestrian/cyclist)"""
        self.pedestrians[psm.user_id] = psm

    def _process_rsa(self, rsa: RoadSideAlert):
        """Process Road Side Alert"""
        # Store in infrastructure nodes
        self.infrastructure_nodes[rsa.alert_id] = {
            'type': 'road_alert',
            'last_update': time.time(),
            'data': rsa
        }

    def _cleanup_stale_vehicles(self, current_time: float, timeout: float = 2.0):
        """Remove stale remote vehicle entries"""
        stale_ids = [
            vid for vid, vehicle in self.remote_vehicles.items()
            if vehicle.is_stale(current_time, timeout)
        ]

        for vid in stale_ids:
            del self.remote_vehicles[vid]

        # Update statistics
        self.stats.remote_vehicles_count = len(self.remote_vehicles)
        self.stats.infrastructure_nodes_count = len(self.infrastructure_nodes)
        self.stats.pedestrians_count = len(self.pedestrians)

    def _update_cooperative_awareness(self, current_time: float):
        """
        Update cooperative awareness based on received messages

        This includes:
        - Collision risk assessment with nearby vehicles
        - Cooperative maneuver planning
        - Platooning coordination
        """
        if not self.current_vehicle_state:
            return

        # Check for nearby vehicles
        for vehicle_id, remote in self.remote_vehicles.items():
            distance = self.current_vehicle_state.position.distance_to(remote.last_state.position)

            # If vehicle is within awareness threshold
            if distance < self.config.min_distance_threshold:
                # Calculate time-to-collision if applicable
                ttc = self._calculate_time_to_collision(
                    self.current_vehicle_state,
                    remote.last_state
                )

                if ttc is not None and ttc < 3.0:  # Less than 3 seconds
                    print(f"[V2X] Warning: Potential collision with {vehicle_id} in {ttc:.1f}s")

    def _calculate_time_to_collision(self, ego: VehicleState, other: VehicleState) -> Optional[float]:
        """
        Calculate time-to-collision between ego vehicle and other vehicle

        Args:
            ego: Ego vehicle state
            other: Other vehicle state

        Returns:
            Time to collision in seconds, or None if no collision predicted
        """
        # Convert heading to radians
        ego_heading_rad = math.radians(ego.heading)
        other_heading_rad = math.radians(other.heading)

        # Calculate velocity vectors
        ego_vx = ego.speed * math.sin(ego_heading_rad)
        ego_vy = ego.speed * math.cos(ego_heading_rad)

        other_vx = other.speed * math.sin(other_heading_rad)
        other_vy = other.speed * math.cos(other_heading_rad)

        # Relative velocity
        rel_vx = other_vx - ego_vx
        rel_vy = other_vy - ego_vy
        rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)

        if rel_speed < 0.1:  # Nearly stationary relative to each other
            return None

        # Calculate distance
        distance = ego.position.distance_to(other.position)

        # Simple TTC calculation (assuming constant velocity)
        # More sophisticated version would account for acceleration
        ttc = distance / rel_speed

        return ttc if ttc > 0 else None

    def get_nearby_vehicles(self, max_distance: float = 150.0) -> List[RemoteVehicle]:
        """
        Get list of nearby vehicles within specified distance

        Args:
            max_distance: Maximum distance in meters

        Returns:
            List of RemoteVehicle objects
        """
        if not self.current_vehicle_state:
            return []

        nearby = []
        for vehicle in self.remote_vehicles.values():
            distance = self.current_vehicle_state.position.distance_to(
                vehicle.last_state.position
            )
            if distance <= max_distance:
                nearby.append(vehicle)

        return nearby

    def get_traffic_signal_state(self, intersection_id: str) -> Optional[SignalPhaseAndTiming]:
        """Get current traffic signal state for intersection"""
        if intersection_id in self.infrastructure_nodes:
            node = self.infrastructure_nodes[intersection_id]
            if node['type'] == 'traffic_signal':
                return node['data']
        return None

    def get_road_alerts(self, max_distance: float = 500.0) -> List[RoadSideAlert]:
        """
        Get nearby road alerts (hazards, construction, etc.)

        Args:
            max_distance: Maximum distance in meters

        Returns:
            List of RoadSideAlert objects
        """
        if not self.current_vehicle_state:
            return []

        alerts = []
        for node_id, node in self.infrastructure_nodes.items():
            if node['type'] == 'road_alert':
                rsa = node['data']
                if rsa.position:
                    distance = self.current_vehicle_state.position.distance_to(rsa.position)
                    if distance <= max_distance:
                        alerts.append(rsa)

        return alerts

    def broadcast_emergency(self, description: str):
        """Broadcast emergency vehicle alert with highest priority"""
        if not self.current_vehicle_state:
            return

        # Create emergency alert
        alert = RoadSideAlert(
            alert_type="emergency_vehicle",
            description=description,
            position=self.current_vehicle_state.position,
            severity=5,
            radius=500.0,
            priority=MessagePriority.EMERGENCY
        )

        self.send_message(alert, MessagePriority.EMERGENCY)
        print(f"[V2X] Emergency broadcast sent: {description}")

    def _generate_private_key(self) -> str:
        """Generate private key for message signing (simplified)"""
        # In production, use proper cryptographic key generation
        return hashlib.sha256(f"{self.vehicle_id}_{time.time()}".encode()).hexdigest()

    def _generate_certificate(self) -> str:
        """Generate certificate for authentication (simplified)"""
        # In production, use proper PKI certificate
        return hashlib.sha256(f"CERT_{self.vehicle_id}".encode()).hexdigest()

    def _sign_message(self, message: V2XMessage) -> str:
        """
        Sign message for security (simplified)

        In production, use proper ECDSA or similar signatures
        """
        message_data = self._serialize_message(message)
        combined = f"{message_data}{self.private_key}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _verify_signature(self, message: V2XMessage) -> bool:
        """
        Verify message signature (simplified)

        In production, verify against certificate authority
        """
        # For now, just check that signature exists and has correct format
        return len(message.signature) == 64  # SHA256 hex digest length

    def _serialize_message(self, message: V2XMessage) -> str:
        """Serialize message to string for signing/transmission"""
        # Simplified serialization
        return json.dumps({
            'protocol': message.protocol.value,
            'seq': message.sequence_number,
            'power': message.transmission_power
        })

    def get_statistics(self) -> V2XStatistics:
        """Get communication statistics"""
        return self.stats

    def visualize_cooperative_awareness(self) -> str:
        """
        Generate text visualization of cooperative awareness

        Returns:
            Text representation of nearby entities
        """
        lines = ["=" * 60]
        lines.append("V2X Cooperative Awareness")
        lines.append("=" * 60)

        lines.append(f"\nRemote Vehicles: {len(self.remote_vehicles)}")
        for vid, vehicle in list(self.remote_vehicles.items())[:5]:  # Show first 5
            distance = 0.0
            if self.current_vehicle_state:
                distance = self.current_vehicle_state.position.distance_to(
                    vehicle.last_state.position
                )
            lines.append(f"  {vid[:8]}: {distance:.1f}m, {vehicle.last_state.speed:.1f}m/s, "
                        f"RSSI: {vehicle.average_rssi:.1f}dBm")

        lines.append(f"\nInfrastructure Nodes: {len(self.infrastructure_nodes)}")
        for node_id, node in list(self.infrastructure_nodes.items())[:3]:
            lines.append(f"  {node_id}: {node['type']}")

        lines.append(f"\nPedestrians: {len(self.pedestrians)}")

        lines.append(f"\nStatistics:")
        lines.append(f"  Messages Sent: {self.stats.messages_sent}")
        lines.append(f"  Messages Received: {self.stats.messages_received}")
        lines.append(f"  Messages Dropped: {self.stats.messages_dropped}")
        lines.append(f"  Avg Latency: {self.stats.average_latency*1000:.1f}ms")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Test V2X communication system"""
    print("Testing V2X Communication System\n")

    # Create configuration
    config = V2XConfig(
        protocol=V2XProtocol.DSRC,
        bsm_frequency=10.0,
        max_range=300.0,
        enable_cooperative_awareness=True
    )

    # Create V2X system for ego vehicle
    ego_v2x = V2XCommunication("EGO_VEHICLE_001", config)

    # Create ego vehicle state
    ego_state = VehicleState(
        vehicle_id="EGO_VEHICLE_001",
        timestamp=time.time(),
        position=Position3D(37.7749, -122.4194, 10.0),  # San Francisco
        heading=90.0,  # East
        speed=15.0,  # 15 m/s ≈ 54 km/h
        acceleration=0.0,
        steering_angle=0.0
    )

    ego_v2x.update_vehicle_state(ego_state)

    # Simulate receiving BSM from another vehicle
    other_state = VehicleState(
        vehicle_id="OTHER_VEHICLE_001",
        timestamp=time.time(),
        position=Position3D(37.7750, -122.4190, 10.0),  # ~40m away
        heading=270.0,  # West
        speed=12.0,
        acceleration=0.0
    )

    other_bsm = BasicSafetyMessage(vehicle_state=other_state)
    other_msg = V2XMessage(
        payload=other_bsm,
        protocol=V2XProtocol.DSRC,
        sequence_number=1
    )

    ego_v2x.receive_message(other_msg, rssi=-75.0)

    # Simulate SPaT message from traffic signal
    spat = SignalPhaseAndTiming(
        intersection_id="INTERSECTION_001",
        signal_groups={
            1: (SignalPhase.GREEN, 15.0),  # Group 1: Green for 15 more seconds
            2: (SignalPhase.RED, 20.0)      # Group 2: Red for 20 more seconds
        }
    )

    spat_msg = V2XMessage(
        payload=spat,
        protocol=V2XProtocol.DSRC,
        sequence_number=2
    )

    ego_v2x.receive_message(spat_msg, rssi=-70.0)

    # Simulate road alert
    alert = RoadSideAlert(
        alert_id="ALERT_001",
        position=Position3D(37.7755, -122.4190, 10.0),
        alert_type="construction",
        description="Road construction ahead - right lane closed",
        severity=3,
        radius=200.0
    )

    alert_msg = V2XMessage(
        payload=alert,
        protocol=V2XProtocol.DSRC,
        sequence_number=3
    )

    ego_v2x.receive_message(alert_msg, rssi=-80.0)

    # Update V2X system
    print("Updating V2X system...")
    ego_v2x.update(0.1)

    # Get nearby vehicles
    nearby = ego_v2x.get_nearby_vehicles(max_distance=200.0)
    print(f"\nNearby vehicles: {len(nearby)}")
    for vehicle in nearby:
        print(f"  {vehicle.vehicle_id}: {vehicle.last_state.speed:.1f} m/s")

    # Get traffic signal state
    signal = ego_v2x.get_traffic_signal_state("INTERSECTION_001")
    if signal:
        print(f"\nTraffic Signal (INTERSECTION_001):")
        for group_id, (phase, time_remaining) in signal.signal_groups.items():
            print(f"  Group {group_id}: {phase.value} ({time_remaining:.1f}s remaining)")

    # Get road alerts
    alerts = ego_v2x.get_road_alerts(max_distance=500.0)
    print(f"\nRoad Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert.alert_type}: {alert.description} (severity: {alert.severity}/5)")

    # Print cooperative awareness
    print("\n" + ego_v2x.visualize_cooperative_awareness())

    # Test emergency broadcast
    print("\nTesting emergency broadcast...")
    ego_v2x.broadcast_emergency("Emergency vehicle approaching from behind")

    # Final statistics
    stats = ego_v2x.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Messages Sent: {stats.messages_sent}")
    print(f"  Messages Received: {stats.messages_received}")
    print(f"  Average Latency: {stats.average_latency*1000:.2f}ms")

    print("\n✓ V2X Communication test complete!")


if __name__ == "__main__":
    main()
