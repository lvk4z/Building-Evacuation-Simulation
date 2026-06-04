from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class ExitDefinition:
    """Description of a physical exit opening in the building shell."""

    name: str
    side: str
    center: np.ndarray
    width: float
    cells: list[tuple[int, int]]


@dataclass(slots=True)
class SpawnZone:
    """Rectangular room region used for initial agent placement."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def area(self) -> float:
        return max(self.x_max - self.x_min, 0.0) * max(self.y_max - self.y_min, 0.0)


class BuildingLayout:
    """Grid-backed baseline building layout.

    The alpha version uses a hand-crafted floor plan with a central corridor,
    seven rooms, four exits, and two columns. The geometry is intentionally
    simple so the core simulation is easy to inspect and extend.
    """

    def __init__(self, width_m: float, height_m: float, cell_size: float) -> None:
        self.width_m = width_m
        self.height_m = height_m
        self.cell_size = cell_size
        self.grid_width = int(round(width_m / cell_size))
        self.grid_height = int(round(height_m / cell_size))

        # True = walkable cell, False = wall / obstacle.
        self.walkable = np.ones((self.grid_height, self.grid_width), dtype=bool)
        self.exits: dict[str, ExitDefinition] = {}
        self.spawn_zones: list[SpawnZone] = []
        self._blocked_centers_cache: np.ndarray | None = None

    @classmethod
    def baseline(cls, width_m: float = 36.0, height_m: float = 20.0, cell_size: float = 0.4) -> "BuildingLayout":
        """Create the baseline ground-floor plan described in the project brief."""

        layout = cls(width_m=width_m, height_m=height_m, cell_size=cell_size)

        # Outer shell is closed by default; exit openings are carved afterwards.
        layout.walkable[0, :] = False
        layout.walkable[-1, :] = False
        layout.walkable[:, 0] = False
        layout.walkable[:, -1] = False

        # Internal room-to-corridor separators with door openings.
        layout._add_horizontal_wall(y=8.0, door_spans=[(5.1, 6.9), (17.1, 18.9), (29.1, 30.9)])
        layout._add_horizontal_wall(y=12.0, door_spans=[(3.6, 5.4), (12.6, 14.4), (21.6, 23.4), (30.6, 32.4)])

        # Room dividers above and below the corridor.
        for x_pos in (9.0, 18.0, 27.0):
            layout._add_vertical_wall(x=x_pos, y_min=12.0, y_max=height_m, door_spans=[])
        for x_pos in (12.0, 24.0):
            layout._add_vertical_wall(x=x_pos, y_min=0.0, y_max=8.0, door_spans=[])

        # Corridor columns.
        layout._fill_rectangle(x_min=13.6, x_max=15.0, y_min=8.9, y_max=11.1, walkable=False)
        layout._fill_rectangle(x_min=21.2, x_max=22.6, y_min=8.9, y_max=11.1, walkable=False)

        # Exits on the shell.
        layout._register_exit(name="E1", side="west", center=10.0, width=2.0)
        layout._register_exit(name="E2", side="east", center=10.0, width=1.2)
        # The north exit is aligned with the top-middle-right room doorway so
        # the baseline path field remains smooth on the coarse alpha grid.
        layout._register_exit(name="E3", side="north", center=22.5, width=1.0)
        layout._register_exit(name="E4", side="south", center=18.0, width=1.8)

        # Room spawn zones. Each zone is inset from the walls to reduce
        # impossible initial overlaps.
        layout.spawn_zones = [
            SpawnZone("top_left_1", 1.0, 8.0, 13.0, 19.0),
            SpawnZone("top_left_2", 10.0, 17.0, 13.0, 19.0),
            SpawnZone("top_right_1", 19.0, 26.0, 13.0, 19.0),
            SpawnZone("top_right_2", 28.0, 35.0, 13.0, 19.0),
            SpawnZone("bottom_left", 1.0, 11.0, 1.0, 7.0),
            SpawnZone("bottom_center", 13.0, 23.0, 1.0, 7.0),
            SpawnZone("bottom_right", 25.0, 35.0, 1.0, 7.0),
        ]

        return layout

    def _fill_rectangle(self, x_min: float, x_max: float, y_min: float, y_max: float, walkable: bool) -> None:
        """Set all cells intersecting the given rectangle to the desired state."""

        x0, x1 = self._x_slice(x_min, x_max)
        y0, y1 = self._y_slice(y_min, y_max)
        self.walkable[y0:y1, x0:x1] = walkable
        self._blocked_centers_cache = None

    def _add_horizontal_wall(self, y: float, door_spans: list[tuple[float, float]]) -> None:
        """Create a one-cell-thick wall line and keep openings at doors."""

        y_index = self.world_to_cell(np.array([self.cell_size, y]))[1]
        self.walkable[y_index, :] = False
        self.walkable[y_index, 0] = False
        self.walkable[y_index, -1] = False
        for x_min, x_max in door_spans:
            start, end = self._x_slice(x_min, x_max)
            self.walkable[y_index, start:end] = True
        self._blocked_centers_cache = None

    def _add_vertical_wall(self, x: float, y_min: float, y_max: float, door_spans: list[tuple[float, float]]) -> None:
        """Create a one-cell-thick wall line between rooms."""

        x_index = self.world_to_cell(np.array([x, self.cell_size]))[0]
        start, end = self._y_slice(y_min, y_max)
        self.walkable[start:end, x_index] = False
        for door_y_min, door_y_max in door_spans:
            door_start, door_end = self._y_slice(door_y_min, door_y_max)
            self.walkable[door_start:door_end, x_index] = True
        self._blocked_centers_cache = None

    def _register_exit(self, name: str, side: str, center: float, width: float) -> None:
        """Carve an opening in the shell and store its metadata."""

        if side in {"west", "east"}:
            y_min = center - width / 2.0
            y_max = center + width / 2.0
            y0, y1 = self._y_slice(y_min, y_max)
            x_index = 0 if side == "west" else self.grid_width - 1
            self.walkable[y0:y1, x_index] = True
            cells = [(x_index, y) for y in range(y0, y1)]
            center_point = np.array([0.0 if side == "west" else self.width_m, center], dtype=float)
        else:
            x_min = center - width / 2.0
            x_max = center + width / 2.0
            x0, x1 = self._x_slice(x_min, x_max)
            y_index = self.grid_height - 1 if side == "north" else 0
            self.walkable[y_index, x0:x1] = True
            cells = [(x, y_index) for x in range(x0, x1)]
            center_point = np.array([center, self.height_m if side == "north" else 0.0], dtype=float)

        self.exits[name] = ExitDefinition(
            name=name,
            side=side,
            center=center_point,
            width=width,
            cells=cells,
        )
        self._blocked_centers_cache = None

    def _x_slice(self, x_min: float, x_max: float) -> tuple[int, int]:
        start = max(0, int(np.floor(x_min / self.cell_size)))
        end = min(self.grid_width, int(np.ceil(x_max / self.cell_size)))
        return start, max(start + 1, end)

    def _y_slice(self, y_min: float, y_max: float) -> tuple[int, int]:
        start = max(0, int(np.floor(y_min / self.cell_size)))
        end = min(self.grid_height, int(np.ceil(y_max / self.cell_size)))
        return start, max(start + 1, end)

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        """Convert a world position to integer grid coordinates."""

        x_index = int(np.clip(point[0] / self.cell_size, 0, self.grid_width - 1))
        y_index = int(np.clip(point[1] / self.cell_size, 0, self.grid_height - 1))
        return x_index, y_index

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """Return the center point of a grid cell in world coordinates."""

        x_index, y_index = cell
        return np.array([(x_index + 0.5) * self.cell_size, (y_index + 0.5) * self.cell_size], dtype=float)

    def is_walkable_cell(self, cell: tuple[int, int]) -> bool:
        """Check whether a grid cell is usable for movement."""

        x_index, y_index = cell
        if x_index < 0 or y_index < 0 or x_index >= self.grid_width or y_index >= self.grid_height:
            return False
        return bool(self.walkable[y_index, x_index])

    def is_walkable_point(self, point: np.ndarray) -> bool:
        """Check whether a world point lies inside a walkable cell."""

        return self.is_walkable_cell(self.world_to_cell(point))

    def get_blocked_cell_centers(self) -> np.ndarray:
        """Return centers of blocked cells for approximate wall repulsion."""

        if self._blocked_centers_cache is None:
            blocked_y, blocked_x = np.where(~self.walkable)
            centers = np.column_stack(
                ((blocked_x + 0.5) * self.cell_size, (blocked_y + 0.5) * self.cell_size)
            )
            self._blocked_centers_cache = centers.astype(float)
        return self._blocked_centers_cache

    def is_at_exit(self, point: np.ndarray, exit_name: str) -> bool:
        """Return whether a point has reached the opening of a given exit."""

        exit_def = self.exits[exit_name]
        # Use a slightly enlarged capture band so agents are absorbed when they
        # reach the opening instead of oscillating against the shell wall.
        half_width = exit_def.width / 2.0 + self.cell_size * 0.9
        capture_depth = self.cell_size * 1.6

        if exit_def.side == "west":
            return point[0] <= capture_depth and abs(point[1] - exit_def.center[1]) <= half_width
        if exit_def.side == "east":
            return point[0] >= self.width_m - capture_depth and abs(point[1] - exit_def.center[1]) <= half_width
        if exit_def.side == "south":
            return point[1] <= capture_depth and abs(point[0] - exit_def.center[0]) <= half_width
        return point[1] >= self.height_m - capture_depth and abs(point[0] - exit_def.center[0]) <= half_width

    def resolve_move(self, old_point: np.ndarray, candidate_point: np.ndarray) -> np.ndarray:
        """Try full motion first, then simple axis-aligned sliding.

        The method keeps the baseline engine stable without requiring a full
        rigid-body collision solver.
        """

        if self.is_walkable_point(candidate_point):
            return candidate_point

        slide_x = np.array([candidate_point[0], old_point[1]], dtype=float)
        if self.is_walkable_point(slide_x):
            return slide_x

        slide_y = np.array([old_point[0], candidate_point[1]], dtype=float)
        if self.is_walkable_point(slide_y):
            return slide_y

        current_cell = self.world_to_cell(old_point)
        walkable_neighbors: list[np.ndarray] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current_cell[0] + dx, current_cell[1] + dy)
                if self.is_walkable_cell(neighbor):
                    walkable_neighbors.append(self.cell_to_world(neighbor))

        if walkable_neighbors:
            target = min(walkable_neighbors, key=lambda point: float(np.linalg.norm(point - candidate_point)))
            direction = target - old_point
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                fallback_point = old_point + direction / norm * min(norm, self.cell_size * 0.45)
                if self.is_walkable_point(fallback_point):
                    return fallback_point

        return old_point.copy()

    def sample_spawn_positions(self, count: int, rng: np.random.Generator, min_distance: float = 0.6) -> list[np.ndarray]:
        """Sample non-overlapping start positions across the room zones."""

        if count <= 0:
            return []

        zone_weights = np.array([zone.area for zone in self.spawn_zones], dtype=float)
        zone_weights /= zone_weights.sum()
        positions: list[np.ndarray] = []
        max_attempts = 50000

        for _ in range(max_attempts):
            if len(positions) >= count:
                break

            zone = self.spawn_zones[int(rng.choice(len(self.spawn_zones), p=zone_weights))]
            point = np.array(
                [rng.uniform(zone.x_min, zone.x_max), rng.uniform(zone.y_min, zone.y_max)],
                dtype=float,
            )

            if not self.is_walkable_point(point):
                continue

            if any(np.linalg.norm(point - existing) < min_distance for existing in positions):
                continue

            positions.append(point)

        if len(positions) != count:
            import warnings
            warnings.warn(
                f"Could only place {len(positions)} / {count} agents. "
                "Proceeding with fewer agents (smoke may have blocked spawn zones).",
                RuntimeWarning,
                stacklevel=2,
            )

        return positions

    def iter_exit_segments(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield short line segments that are useful for plotting exits."""

        for _, start, end in self.iter_named_exit_segments():
            yield start, end

    def iter_named_exit_segments(self) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
        """Yield exit names together with plot-friendly boundary segments."""

        for exit_def in self.exits.values():
            if exit_def.side in {"west", "east"}:
                x = 0.0 if exit_def.side == "west" else self.width_m
                start = np.array([x, exit_def.center[1] - exit_def.width / 2.0])
                end = np.array([x, exit_def.center[1] + exit_def.width / 2.0])
            else:
                y = 0.0 if exit_def.side == "south" else self.height_m
                start = np.array([exit_def.center[0] - exit_def.width / 2.0, y])
                end = np.array([exit_def.center[0] + exit_def.width / 2.0, y])
            yield exit_def.name, start, end
