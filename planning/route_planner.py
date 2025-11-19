"""
Lane Graph & Global Route Planning Module

This module provides global route planning on lane-level graphs:
- Lane graph construction and representation
- A* pathfinding on lane graph
- Dijkstra for shortest path
- Multi-criteria route optimization
- Lane-level turn instructions
- Route alternatives generation
- Intersection navigation
- Lane change recommendations
- Route replanning on blockage
- Route visualization

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from collections import deque
import heapq
import math


class TurnDirection(Enum):
    """Turn directions at intersections"""
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    SLIGHT_LEFT = "slight_left"
    SLIGHT_RIGHT = "slight_right"
    SHARP_LEFT = "sharp_left"
    SHARP_RIGHT = "sharp_right"
    U_TURN = "u_turn"


class RouteStatus(Enum):
    """Route planning status"""
    SUCCESS = "success"
    NO_PATH = "no_path"
    START_INVALID = "start_invalid"
    GOAL_INVALID = "goal_invalid"
    PLANNING_FAILED = "planning_failed"


@dataclass
class LaneGraphNode:
    """Node in lane graph (represents a lane segment)"""
    node_id: int
    lane_id: int  # Corresponding lane ID in HD map
    center_point: Tuple[float, float]  # Representative point
    length: float  # Lane segment length
    speed_limit: float  # m/s
    lane_type: str  # "driving", "bus", "bike", etc.

    # Graph connectivity
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)

    # Routing attributes
    cost_to_traverse: float = 0.0  # Base cost
    is_blocked: bool = False


@dataclass
class LaneGraphEdge:
    """Edge in lane graph (lane transition)"""
    from_node: int
    to_node: int
    turn_direction: TurnDirection
    cost: float  # Transition cost
    is_lane_change: bool = False
    change_direction: Optional[str] = None  # "left" or "right"


@dataclass
class RouteSegment:
    """Segment of a route"""
    node_id: int
    lane_id: int
    distance: float  # Distance to travel in this segment
    instruction: str  # Human-readable instruction
    turn_direction: Optional[TurnDirection] = None
    cumulative_distance: float = 0.0


@dataclass
class Route:
    """Complete route from start to goal"""
    segments: List[RouteSegment]
    total_distance: float
    estimated_time: float  # seconds
    status: RouteStatus
    cost: float  # Total route cost
    alternative_rank: int = 0  # 0 = primary, 1+ = alternatives


class LaneGraph:
    """
    Lane-Level Road Graph

    Represents road network at lane granularity for routing
    """

    def __init__(self):
        """Initialize lane graph"""
        self.nodes: Dict[int, LaneGraphNode] = {}
        self.edges: Dict[Tuple[int, int], LaneGraphEdge] = {}

        # Spatial index for nearest node queries
        self.node_positions: Dict[int, Tuple[float, float]] = {}

    def add_node(self, node: LaneGraphNode):
        """Add node to graph"""
        self.nodes[node.node_id] = node
        self.node_positions[node.node_id] = node.center_point

    def add_edge(self, edge: LaneGraphEdge):
        """Add edge to graph"""
        key = (edge.from_node, edge.to_node)
        self.edges[key] = edge

        # Update node connectivity
        if edge.from_node in self.nodes:
            if edge.to_node not in self.nodes[edge.from_node].successors:
                self.nodes[edge.from_node].successors.append(edge.to_node)

        if edge.to_node in self.nodes:
            if edge.from_node not in self.nodes[edge.to_node].predecessors:
                self.nodes[edge.to_node].predecessors.append(edge.from_node)

    def get_nearest_node(self, position: Tuple[float, float]) -> Optional[int]:
        """Find nearest graph node to position"""
        if not self.node_positions:
            return None

        min_distance = float('inf')
        nearest_node_id = None

        for node_id, node_pos in self.node_positions.items():
            distance = math.sqrt(
                (position[0] - node_pos[0])**2 +
                (position[1] - node_pos[1])**2
            )

            if distance < min_distance:
                min_distance = distance
                nearest_node_id = node_id

        return nearest_node_id

    def get_edge(self, from_node: int, to_node: int) -> Optional[LaneGraphEdge]:
        """Get edge between two nodes"""
        return self.edges.get((from_node, to_node))

    def set_node_blocked(self, node_id: int, blocked: bool):
        """Block/unblock a node"""
        if node_id in self.nodes:
            self.nodes[node_id].is_blocked = blocked


class RoutePlanner:
    """
    Global Route Planner

    Features:
    - A* and Dijkstra pathfinding
    - Multi-criteria optimization
    - Alternative route generation
    - Lane-level instructions
    - Route replanning
    """

    def __init__(self, lane_graph: LaneGraph):
        """
        Initialize route planner

        Args:
            lane_graph: Lane graph for routing
        """
        self.graph = lane_graph

        # Routing preferences
        self.prefer_highway = True
        self.avoid_tolls = False
        self.minimize_lane_changes = True

        # Statistics
        self.total_routes_planned = 0
        self.successful_routes = 0
        self.failed_routes = 0

    def plan_route(
        self,
        start_position: Tuple[float, float],
        goal_position: Tuple[float, float],
        num_alternatives: int = 0
    ) -> List[Route]:
        """
        Plan route from start to goal

        Args:
            start_position: Start position (x, y)
            goal_position: Goal position (x, y)
            num_alternatives: Number of alternative routes to generate

        Returns:
            List of routes (primary + alternatives)
        """
        self.total_routes_planned += 1

        # Find nearest nodes
        start_node = self.graph.get_nearest_node(start_position)
        goal_node = self.graph.get_nearest_node(goal_position)

        if start_node is None:
            return [Route([], 0.0, 0.0, RouteStatus.START_INVALID, 0.0)]

        if goal_node is None:
            return [Route([], 0.0, 0.0, RouteStatus.GOAL_INVALID, 0.0)]

        # Plan primary route using A*
        primary_route = self._astar_search(start_node, goal_node)

        if primary_route.status != RouteStatus.SUCCESS:
            self.failed_routes += 1
            return [primary_route]

        primary_route.alternative_rank = 0
        routes = [primary_route]

        # Generate alternatives if requested
        if num_alternatives > 0:
            alternatives = self._generate_alternatives(
                start_node, goal_node, primary_route, num_alternatives
            )
            routes.extend(alternatives)

        self.successful_routes += 1
        return routes

    def _astar_search(
        self,
        start_node_id: int,
        goal_node_id: int
    ) -> Route:
        """A* pathfinding on lane graph"""

        # Priority queue: (f_score, node_id)
        open_set = [(0.0, start_node_id)]

        # Best known cost to reach each node
        g_score = {start_node_id: 0.0}

        # Came from tracking for path reconstruction
        came_from: Dict[int, int] = {}

        # Nodes already evaluated
        closed_set: Set[int] = set()

        goal_node = self.graph.nodes[goal_node_id]
        goal_pos = goal_node.center_point

        while open_set:
            # Get node with lowest f_score
            current_f, current_id = heapq.heappop(open_set)

            # Check if goal reached
            if current_id == goal_node_id:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current_id)
                return self._path_to_route(path)

            # Skip if already evaluated
            if current_id in closed_set:
                continue

            closed_set.add(current_id)

            current_node = self.graph.nodes[current_id]
            current_g = g_score[current_id]

            # Expand neighbors
            for successor_id in current_node.successors:
                if successor_id in closed_set:
                    continue

                successor_node = self.graph.nodes[successor_id]

                # Skip blocked nodes
                if successor_node.is_blocked:
                    continue

                # Calculate tentative g_score
                edge = self.graph.get_edge(current_id, successor_id)

                if edge:
                    edge_cost = edge.cost
                else:
                    # Default cost based on distance
                    edge_cost = successor_node.cost_to_traverse

                # Add penalties for lane changes
                if edge and edge.is_lane_change and self.minimize_lane_changes:
                    edge_cost += 10.0

                tentative_g = current_g + edge_cost

                # Check if this path is better
                if successor_id not in g_score or tentative_g < g_score[successor_id]:
                    # Update best path
                    came_from[successor_id] = current_id
                    g_score[successor_id] = tentative_g

                    # Calculate heuristic (Euclidean distance to goal)
                    h_score = math.sqrt(
                        (successor_node.center_point[0] - goal_pos[0])**2 +
                        (successor_node.center_point[1] - goal_pos[1])**2
                    )

                    f_score = tentative_g + h_score

                    heapq.heappush(open_set, (f_score, successor_id))

        # No path found
        return Route([], 0.0, 0.0, RouteStatus.NO_PATH, 0.0)

    def _reconstruct_path(
        self,
        came_from: Dict[int, int],
        current_id: int
    ) -> List[int]:
        """Reconstruct path from came_from map"""
        path = [current_id]

        while current_id in came_from:
            current_id = came_from[current_id]
            path.append(current_id)

        path.reverse()
        return path

    def _path_to_route(self, path: List[int]) -> Route:
        """Convert node path to route with instructions"""
        if not path:
            return Route([], 0.0, 0.0, RouteStatus.NO_PATH, 0.0)

        segments = []
        total_distance = 0.0
        cumulative_distance = 0.0

        for i in range(len(path)):
            node_id = path[i]
            node = self.graph.nodes[node_id]

            # Determine turn direction
            turn_direction = None
            if i > 0:
                prev_node_id = path[i-1]
                edge = self.graph.get_edge(prev_node_id, node_id)
                if edge:
                    turn_direction = edge.turn_direction

            # Generate instruction
            instruction = self._generate_instruction(
                node, turn_direction, i == 0, i == len(path) - 1
            )

            # Calculate distance
            distance = node.length
            total_distance += distance

            segment = RouteSegment(
                node_id=node_id,
                lane_id=node.lane_id,
                distance=distance,
                instruction=instruction,
                turn_direction=turn_direction,
                cumulative_distance=cumulative_distance
            )

            segments.append(segment)
            cumulative_distance += distance

        # Estimate time
        avg_speed = 15.0  # m/s (54 km/h)
        estimated_time = total_distance / avg_speed

        return Route(
            segments=segments,
            total_distance=total_distance,
            estimated_time=estimated_time,
            status=RouteStatus.SUCCESS,
            cost=total_distance
        )

    def _generate_instruction(
        self,
        node: LaneGraphNode,
        turn: Optional[TurnDirection],
        is_start: bool,
        is_end: bool
    ) -> str:
        """Generate human-readable instruction"""
        if is_start:
            return "Start route"

        if is_end:
            return "Arrive at destination"

        if turn:
            if turn == TurnDirection.STRAIGHT:
                return "Continue straight"
            elif turn == TurnDirection.LEFT:
                return "Turn left"
            elif turn == TurnDirection.RIGHT:
                return "Turn right"
            elif turn == TurnDirection.SLIGHT_LEFT:
                return "Bear left"
            elif turn == TurnDirection.SLIGHT_RIGHT:
                return "Bear right"
            elif turn == TurnDirection.SHARP_LEFT:
                return "Sharp left turn"
            elif turn == TurnDirection.SHARP_RIGHT:
                return "Sharp right turn"
            elif turn == TurnDirection.U_TURN:
                return "Make U-turn"

        return f"Continue on {node.lane_type} lane"

    def _generate_alternatives(
        self,
        start_node: int,
        goal_node: int,
        primary_route: Route,
        num_alternatives: int
    ) -> List[Route]:
        """Generate alternative routes"""
        alternatives = []

        # Extract primary path nodes
        primary_nodes = {seg.node_id for seg in primary_route.segments}

        # Try blocking nodes from primary route and replanning
        for i in range(min(num_alternatives, len(primary_route.segments) // 3)):
            # Block middle nodes from primary route
            block_idx = len(primary_route.segments) // 2 + i

            if block_idx < len(primary_route.segments):
                node_to_block = primary_route.segments[block_idx].node_id

                # Temporarily block node
                original_state = self.graph.nodes[node_to_block].is_blocked
                self.graph.set_node_blocked(node_to_block, True)

                # Plan alternative
                alt_route = self._astar_search(start_node, goal_node)

                # Restore original state
                self.graph.set_node_blocked(node_to_block, original_state)

                # Check if route is sufficiently different
                if alt_route.status == RouteStatus.SUCCESS:
                    alt_nodes = {seg.node_id for seg in alt_route.segments}
                    overlap = len(primary_nodes & alt_nodes) / len(primary_nodes)

                    if overlap < 0.7:  # Less than 70% overlap
                        alt_route.alternative_rank = i + 1
                        alternatives.append(alt_route)

        return alternatives

    def replan_around_blockage(
        self,
        current_route: Route,
        blocked_node_id: int,
        current_segment_idx: int
    ) -> Optional[Route]:
        """
        Replan route when encountering blockage

        Args:
            current_route: Current route being followed
            blocked_node_id: Node that is now blocked
            current_segment_idx: Current position in route

        Returns:
            New route or None if replanning fails
        """
        # Mark node as blocked
        self.graph.set_node_blocked(blocked_node_id, True)

        # Get current and goal nodes
        current_node = current_route.segments[current_segment_idx].node_id
        goal_node = current_route.segments[-1].node_id

        # Replan from current to goal
        new_route = self._astar_search(current_node, goal_node)

        if new_route.status == RouteStatus.SUCCESS:
            # Prepend completed segments
            completed_segments = current_route.segments[:current_segment_idx]
            new_route.segments = completed_segments + new_route.segments

            # Recalculate totals
            new_route.total_distance = sum(s.distance for s in new_route.segments)
            avg_speed = 15.0
            new_route.estimated_time = new_route.total_distance / avg_speed

            return new_route

        return None

    def get_lane_change_advisory(
        self,
        current_route: Route,
        current_segment_idx: int,
        lookahead_segments: int = 5
    ) -> Optional[str]:
        """
        Get lane change advisory for upcoming maneuvers

        Returns: "left", "right", or None
        """
        # Look ahead in route
        end_idx = min(current_segment_idx + lookahead_segments,
                     len(current_route.segments))

        for i in range(current_segment_idx + 1, end_idx):
            segment = current_route.segments[i]

            if segment.turn_direction:
                if segment.turn_direction in [TurnDirection.LEFT,
                                             TurnDirection.SLIGHT_LEFT,
                                             TurnDirection.SHARP_LEFT]:
                    return "left"
                elif segment.turn_direction in [TurnDirection.RIGHT,
                                               TurnDirection.SLIGHT_RIGHT,
                                               TurnDirection.SHARP_RIGHT]:
                    return "right"

        return None

    def visualize_route(
        self,
        route: Route,
        size: Tuple[int, int] = (800, 800)
    ) -> np.ndarray:
        """Visualize route on map"""
        img = np.zeros((*size, 3), dtype=np.uint8)
        w, h = size

        if not route or not route.segments:
            return img

        # Draw all graph nodes (gray)
        for node in self.graph.nodes.values():
            px = int(w/2 + node.center_point[0] * 5)
            py = int(h/2 - node.center_point[1] * 5)

            if 0 <= px < w and 0 <= py < h:
                color = (50, 50, 50) if not node.is_blocked else (0, 0, 100)
                cv2.circle(img, (px, py), 3, color, -1)

        # Draw route segments (bright)
        for i, segment in enumerate(route.segments):
            node = self.graph.nodes[segment.node_id]
            px = int(w/2 + node.center_point[0] * 5)
            py = int(h/2 - node.center_point[1] * 5)

            if 0 <= px < w and 0 <= py < h:
                # Color based on progress
                progress = i / len(route.segments)
                color = (
                    int(255 * (1 - progress)),
                    int(255 * progress),
                    200
                )
                cv2.circle(img, (px, py), 6, color, -1)

                # Draw connection to next segment
                if i < len(route.segments) - 1:
                    next_node = self.graph.nodes[route.segments[i+1].node_id]
                    next_px = int(w/2 + next_node.center_point[0] * 5)
                    next_py = int(h/2 - next_node.center_point[1] * 5)

                    if 0 <= next_px < w and 0 <= next_py < h:
                        cv2.line(img, (px, py), (next_px, next_py), color, 3)

        # Draw info
        info_lines = [
            f"Route: {len(route.segments)} segments",
            f"Distance: {route.total_distance:.0f}m",
            f"Time: {route.estimated_time/60:.1f} min",
            f"Status: {route.status.value}",
            f"Alt Rank: {route.alternative_rank}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.putText(img, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return img

    def get_statistics(self) -> Dict[str, Any]:
        """Get route planner statistics"""
        success_rate = (self.successful_routes / self.total_routes_planned * 100
                       if self.total_routes_planned > 0 else 0.0)

        return {
            'total_routes_planned': self.total_routes_planned,
            'successful_routes': self.successful_routes,
            'failed_routes': self.failed_routes,
            'success_rate': success_rate,
            'graph_nodes': len(self.graph.nodes),
            'graph_edges': len(self.graph.edges)
        }
