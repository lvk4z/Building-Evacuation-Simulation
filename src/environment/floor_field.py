from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from src.environment.building import BuildingLayout


class StaticFloorField:
    """Static geodesic distance fields computed on the building grid.

    Each exit gets its own distance map. Agents then follow the local descent
    direction of the chosen map, which is enough for a baseline simulation
    without dynamic rerouting.
    """

    def __init__(self, layout: BuildingLayout) -> None:
        self.layout = layout
        self.distance_maps: dict[str, np.ndarray] = {
            name: self._compute_distance_map(exit_def.cells)
            for name, exit_def in layout.exits.items()
        }

    def _compute_distance_map(self, exit_cells: Iterable[tuple[int, int]]) -> np.ndarray:
        distances = np.full((self.layout.grid_height, self.layout.grid_width), np.inf, dtype=float)
        queue: list[tuple[float, int, int]] = []

        for x_index, y_index in exit_cells:
            distances[y_index, x_index] = 0.0
            heapq.heappush(queue, (0.0, x_index, y_index))

        neighbor_offsets = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, np.sqrt(2.0)),
            (-1, 1, np.sqrt(2.0)),
            (1, -1, np.sqrt(2.0)),
            (1, 1, np.sqrt(2.0)),
        ]

        while queue:
            current_distance, x_index, y_index = heapq.heappop(queue)
            if current_distance > distances[y_index, x_index]:
                continue

            for dx, dy, step_cost in neighbor_offsets:
                next_cell = (x_index + dx, y_index + dy)
                if not self.layout.is_walkable_cell(next_cell):
                    continue

                nx, ny = next_cell
                new_distance = current_distance + step_cost * self.layout.cell_size
                if new_distance < distances[ny, nx]:
                    distances[ny, nx] = new_distance
                    heapq.heappush(queue, (new_distance, nx, ny))

        return distances

    def best_exit(self, point: np.ndarray) -> str:
        """Select the exit with smallest static geodesic distance."""

        x_index, y_index = self.layout.world_to_cell(point)
        best_name = min(
            self.distance_maps,
            key=lambda name: self.distance_maps[name][y_index, x_index],
        )
        return best_name

    def direction_for_exit(self, point: np.ndarray, exit_name: str) -> np.ndarray:
        """Return a normalized descent direction of the selected floor field."""

        x_index, y_index = self.layout.world_to_cell(point)
        distances = self.distance_maps[exit_name]
        current_distance = distances[y_index, x_index]

        if not np.isfinite(current_distance):
            # Fallback for points that somehow land on an invalid cell.
            direction = self.layout.exits[exit_name].center - point
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 1e-8 else np.zeros(2, dtype=float)

        best_cell = (x_index, y_index)
        best_distance = current_distance
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                candidate = (x_index + dx, y_index + dy)
                if not self.layout.is_walkable_cell(candidate):
                    continue
                nx, ny = candidate
                candidate_distance = distances[ny, nx]
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_cell = candidate

        if best_cell == (x_index, y_index):
            # When there is no better neighbor, head directly to the exit center.
            direction = self.layout.exits[exit_name].center - point
        else:
            direction = self.layout.cell_to_world(best_cell) - point

        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-8 else np.zeros(2, dtype=float)
