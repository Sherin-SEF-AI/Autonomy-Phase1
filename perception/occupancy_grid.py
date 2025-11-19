"""
Occupancy Grid Mapping Module

This module provides 2D/3D occupancy grid mapping for environment representation:
- Dynamic occupancy grid generation
- Multi-sensor fusion into grid
- Probabilistic occupancy updates (Bayesian)
- Grid decay for dynamic objects
- Cost map generation for planning
- Free space detection
- Grid serialization and persistence
- Real-time visualization

Author: AV Perception System
Version: 1.3.0
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import math


class CellState(Enum):
    """Occupancy cell states"""
    FREE = 0
    OCCUPIED = 1
    UNKNOWN = 2


@dataclass
class OccupancyGridConfig:
    """Configuration for occupancy grid"""
    # Grid dimensions
    width: float = 100.0  # meters
    height: float = 100.0  # meters
    resolution: float = 0.2  # meters per cell
    origin_x: float = 0.0  # Grid origin in world coords
    origin_y: float = 0.0

    # Probabilistic parameters
    prob_occupied: float = 0.7  # P(occupied | sensor detects obstacle)
    prob_free: float = 0.3  # P(free | sensor sees through)
    prob_prior: float = 0.5  # Prior probability (unknown)

    # Log-odds parameters for Bayesian update
    log_odds_occupied: float = 0.85  # log(P/(1-P)) for occupied
    log_odds_free: float = -0.4  # log(P/(1-P)) for free
    log_odds_min: float = -2.0  # Clamp min
    log_odds_max: float = 3.5  # Clamp max

    # Decay for dynamic objects
    enable_decay: bool = True
    decay_rate: float = 0.95  # Decay factor per update (0.95 = 5% decay)

    # Cost map generation
    inflation_radius: float = 1.0  # meters to inflate obstacles
    cost_scaling_factor: float = 10.0


class OccupancyGrid:
    """
    2D Occupancy Grid Map

    Uses log-odds representation for efficient Bayesian updates
    Supports multi-sensor fusion and dynamic object handling
    """

    def __init__(self, config: Optional[OccupancyGridConfig] = None):
        """
        Initialize occupancy grid

        Args:
            config: Grid configuration
        """
        self.config = config or OccupancyGridConfig()

        # Calculate grid size
        self.cols = int(self.config.width / self.config.resolution)
        self.rows = int(self.config.height / self.config.resolution)

        # Log-odds grid for efficient updates
        self.log_odds_grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Timestamps for decay
        self.last_update_time = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Cost map (computed on demand)
        self.cost_map: Optional[np.ndarray] = None
        self.cost_map_dirty = True

        # Statistics
        self.total_updates = 0
        self.total_occupied_cells = 0
        self.total_free_cells = 0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices

        Args:
            x, y: World coordinates (meters)

        Returns:
            (row, col) grid indices
        """
        col = int((x - self.config.origin_x) / self.config.resolution)
        row = int((y - self.config.origin_y) / self.config.resolution)
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates

        Args:
            row, col: Grid indices

        Returns:
            (x, y) world coordinates (center of cell)
        """
        x = col * self.config.resolution + self.config.origin_x + self.config.resolution / 2
        y = row * self.config.resolution + self.config.origin_y + self.config.resolution / 2
        return (x, y)

    def is_valid_cell(self, row: int, col: int) -> bool:
        """Check if cell indices are valid"""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def update_from_point_cloud(
        self,
        points: np.ndarray,
        sensor_position: Tuple[float, float],
        timestamp: float = 0.0
    ):
        """
        Update grid from point cloud

        Args:
            points: Nx2 or Nx3 array of points (x, y, [z])
            sensor_position: (x, y) position of sensor in world coords
            timestamp: Current timestamp for decay
        """
        self.total_updates += 1

        sensor_row, sensor_col = self.world_to_grid(*sensor_position)

        # Process each point
        for point in points:
            point_x, point_y = point[0], point[1]
            point_row, point_col = self.world_to_grid(point_x, point_y)

            if not self.is_valid_cell(point_row, point_col):
                continue

            # Ray tracing: mark cells along ray as free, endpoint as occupied
            ray_cells = self._bresenham_line(
                sensor_row, sensor_col,
                point_row, point_col
            )

            # Update free cells along ray
            for cell_row, cell_col in ray_cells[:-1]:
                if self.is_valid_cell(cell_row, cell_col):
                    self._update_cell_free(cell_row, cell_col, timestamp)

            # Update occupied cell at endpoint
            if self.is_valid_cell(point_row, point_col):
                self._update_cell_occupied(point_row, point_col, timestamp)

        self.cost_map_dirty = True

    def update_from_detections(
        self,
        detections: List[Dict[str, Any]],
        sensor_position: Tuple[float, float],
        timestamp: float = 0.0
    ):
        """
        Update grid from object detections

        Args:
            detections: List of detections with 'position' and 'size'
            sensor_position: Sensor position
            timestamp: Current timestamp
        """
        for detection in detections:
            position = detection.get('position', (0, 0))
            size = detection.get('size', (2.0, 2.0))  # width, height

            # Mark cells in object footprint as occupied
            self._mark_rectangle_occupied(
                position[0], position[1],
                size[0], size[1],
                timestamp
            )

        self.cost_map_dirty = True

    def _mark_rectangle_occupied(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        timestamp: float
    ):
        """Mark rectangular region as occupied"""
        # Calculate grid bounds
        x_min = center_x - width / 2
        x_max = center_x + width / 2
        y_min = center_y - height / 2
        y_max = center_y + height / 2

        row_min, col_min = self.world_to_grid(x_min, y_min)
        row_max, col_max = self.world_to_grid(x_max, y_max)

        # Clamp to grid bounds
        row_min = max(0, row_min)
        row_max = min(self.rows - 1, row_max)
        col_min = max(0, col_min)
        col_max = min(self.cols - 1, col_max)

        # Mark cells as occupied
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                self._update_cell_occupied(row, col, timestamp)

    def _update_cell_occupied(self, row: int, col: int, timestamp: float):
        """Update cell as occupied using log-odds"""
        if self.config.enable_decay:
            self._apply_decay(row, col, timestamp)

        self.log_odds_grid[row, col] += self.config.log_odds_occupied
        self.log_odds_grid[row, col] = np.clip(
            self.log_odds_grid[row, col],
            self.config.log_odds_min,
            self.config.log_odds_max
        )
        self.last_update_time[row, col] = timestamp

        if self.get_cell_probability(row, col) > 0.5:
            self.total_occupied_cells += 1

    def _update_cell_free(self, row: int, col: int, timestamp: float):
        """Update cell as free using log-odds"""
        if self.config.enable_decay:
            self._apply_decay(row, col, timestamp)

        self.log_odds_grid[row, col] += self.config.log_odds_free
        self.log_odds_grid[row, col] = np.clip(
            self.log_odds_grid[row, col],
            self.config.log_odds_min,
            self.config.log_odds_max
        )
        self.last_update_time[row, col] = timestamp

        if self.get_cell_probability(row, col) < 0.5:
            self.total_free_cells += 1

    def _apply_decay(self, row: int, col: int, timestamp: float):
        """Apply temporal decay to cell (for dynamic objects)"""
        if self.last_update_time[row, col] > 0:
            # Decay towards prior (0 in log-odds space)
            self.log_odds_grid[row, col] *= self.config.decay_rate

    def get_cell_probability(self, row: int, col: int) -> float:
        """
        Get occupancy probability for cell

        Returns probability in [0, 1]
        """
        log_odds = self.log_odds_grid[row, col]
        # Convert log-odds to probability: P = 1 / (1 + exp(-log_odds))
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        return prob

    def get_cell_state(self, row: int, col: int) -> CellState:
        """Get discrete cell state"""
        prob = self.get_cell_probability(row, col)

        if prob > 0.6:
            return CellState.OCCUPIED
        elif prob < 0.4:
            return CellState.FREE
        else:
            return CellState.UNKNOWN

    def is_occupied(self, x: float, y: float, threshold: float = 0.6) -> bool:
        """Check if world position is occupied"""
        row, col = self.world_to_grid(x, y)
        if not self.is_valid_cell(row, col):
            return False

        prob = self.get_cell_probability(row, col)
        return prob >= threshold

    def get_cost_map(self, recompute: bool = False) -> np.ndarray:
        """
        Generate cost map with obstacle inflation

        Args:
            recompute: Force recomputation even if cached

        Returns:
            Cost map where 0=free, 255=obstacle, gradual inflation between
        """
        if not self.cost_map_dirty and not recompute and self.cost_map is not None:
            return self.cost_map

        # Initialize cost map
        cost_map = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Mark occupied cells with max cost
        prob_grid = 1.0 / (1.0 + np.exp(-self.log_odds_grid))
        occupied_mask = prob_grid > 0.6
        cost_map[occupied_mask] = 255

        # Inflate obstacles
        if self.config.inflation_radius > 0:
            inflation_cells = int(self.config.inflation_radius / self.config.resolution)

            if inflation_cells > 0:
                # Use distance transform for efficient inflation
                obstacle_mask = (cost_map == 255).astype(np.uint8)
                distances = cv2.distanceTransform(
                    1 - obstacle_mask,
                    cv2.DIST_L2,
                    cv2.DIST_MASK_PRECISE
                )

                # Apply cost gradient
                inflation_cost = np.clip(
                    (inflation_cells - distances) / inflation_cells * 254,
                    0, 254
                ).astype(np.uint8)

                # Combine with obstacles
                cost_map = np.maximum(cost_map, inflation_cost)

        self.cost_map = cost_map
        self.cost_map_dirty = False

        return cost_map

    def _bresenham_line(
        self,
        r0: int, c0: int,
        r1: int, c1: int
    ) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing

        Returns list of (row, col) cells along line
        """
        cells = []

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dr - dc

        r, c = r0, c0

        while True:
            cells.append((r, c))

            if r == r1 and c == c1:
                break

            e2 = 2 * err

            if e2 > -dc:
                err -= dc
                r += sr

            if e2 < dr:
                err += dr
                c += sc

        return cells

    def clear_region(self, x_min: float, y_min: float, x_max: float, y_max: float):
        """Clear occupancy in rectangular region"""
        row_min, col_min = self.world_to_grid(x_min, y_min)
        row_max, col_max = self.world_to_grid(x_max, y_max)

        row_min = max(0, row_min)
        row_max = min(self.rows - 1, row_max)
        col_min = max(0, col_min)
        col_max = min(self.cols - 1, col_max)

        self.log_odds_grid[row_min:row_max+1, col_min:col_max+1] = 0.0
        self.cost_map_dirty = True

    def reset(self):
        """Reset entire grid"""
        self.log_odds_grid.fill(0.0)
        self.last_update_time.fill(0.0)
        self.cost_map = None
        self.cost_map_dirty = True
        self.total_updates = 0
        self.total_occupied_cells = 0
        self.total_free_cells = 0

    def get_free_space_polygon(self) -> List[Tuple[float, float]]:
        """
        Extract free space as polygon

        Returns list of (x, y) points forming free space boundary
        """
        # Get probability grid
        prob_grid = 1.0 / (1.0 + np.exp(-self.log_odds_grid))
        free_mask = (prob_grid < 0.4).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            free_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Convert to world coordinates
        world_points = []
        for point in largest_contour[:, 0, :]:
            col, row = point[0], point[1]
            x, y = self.grid_to_world(row, col)
            world_points.append((x, y))

        return world_points

    def visualize(
        self,
        show_cost_map: bool = False,
        vehicle_position: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Visualize occupancy grid

        Args:
            show_cost_map: Show cost map instead of occupancy
            vehicle_position: Optional vehicle position to draw

        Returns:
            Visualization image
        """
        if show_cost_map:
            # Show cost map
            cost_map = self.get_cost_map()
            vis = cv2.applyColorMap(255 - cost_map, cv2.COLORMAP_JET)
        else:
            # Show occupancy probabilities
            prob_grid = 1.0 / (1.0 + np.exp(-self.log_odds_grid))

            # Map to grayscale: 0 (black) = occupied, 255 (white) = free, 128 = unknown
            vis_gray = ((1.0 - prob_grid) * 255).astype(np.uint8)
            vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

        # Draw vehicle if provided
        if vehicle_position:
            row, col = self.world_to_grid(*vehicle_position)
            if self.is_valid_cell(row, col):
                cv2.circle(vis, (col, row), 5, (0, 0, 255), -1)

                # Draw heading indicator
                cv2.circle(vis, (col, row - 8), 2, (0, 0, 255), -1)

        # Draw grid lines every 5 meters
        grid_spacing = int(5.0 / self.config.resolution)
        for i in range(0, self.cols, grid_spacing):
            cv2.line(vis, (i, 0), (i, self.rows), (100, 100, 100), 1)
        for i in range(0, self.rows, grid_spacing):
            cv2.line(vis, (0, i), (self.cols, i), (100, 100, 100), 1)

        # Add info text
        info_lines = [
            f"Resolution: {self.config.resolution:.2f}m",
            f"Size: {self.config.width:.0f}x{self.config.height:.0f}m",
            f"Updates: {self.total_updates}",
            f"Occupied: {self.total_occupied_cells}",
            f"Free: {self.total_free_cells}"
        ]

        y_offset = 20
        for line in info_lines:
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 15

        return vis

    def save_to_file(self, filename: str):
        """Save grid to file"""
        np.savez_compressed(
            filename,
            log_odds_grid=self.log_odds_grid,
            last_update_time=self.last_update_time,
            config_width=self.config.width,
            config_height=self.config.height,
            config_resolution=self.config.resolution,
            config_origin_x=self.config.origin_x,
            config_origin_y=self.config.origin_y
        )

    def load_from_file(self, filename: str):
        """Load grid from file"""
        data = np.load(filename)

        self.log_odds_grid = data['log_odds_grid']
        self.last_update_time = data['last_update_time']
        self.config.width = float(data['config_width'])
        self.config.height = float(data['config_height'])
        self.config.resolution = float(data['config_resolution'])
        self.config.origin_x = float(data['config_origin_x'])
        self.config.origin_y = float(data['config_origin_y'])

        self.rows, self.cols = self.log_odds_grid.shape
        self.cost_map_dirty = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get occupancy grid statistics"""
        prob_grid = 1.0 / (1.0 + np.exp(-self.log_odds_grid))

        occupied_count = np.sum(prob_grid > 0.6)
        free_count = np.sum(prob_grid < 0.4)
        unknown_count = self.rows * self.cols - occupied_count - free_count

        return {
            'total_updates': self.total_updates,
            'grid_size': (self.rows, self.cols),
            'resolution': self.config.resolution,
            'coverage_area': self.config.width * self.config.height,
            'occupied_cells': int(occupied_count),
            'free_cells': int(free_count),
            'unknown_cells': int(unknown_count),
            'occupied_percentage': float(occupied_count / (self.rows * self.cols) * 100),
            'free_percentage': float(free_count / (self.rows * self.cols) * 100)
        }
