"""Simplified smoke / fire model for building evacuation scenarios.

The fire starts in one cell and smoke spreads outward via a BFS-like
diffusion with a configurable spread radius per simulation step.  Cells
with smoke density above a visibility threshold are treated as impassable
by the floor-field BFS, forcing agents to seek alternative exits.

Usage::

    smoke = SmokeLayer(layout, origin=(18.0, 4.0), spread_rate=0.4)
    # Call each model step:
    smoke.step(current_time=t, dt=dt)
    blocked_cells = smoke.blocked_cells()   # set of (x, y) cell tuples
    smoke.apply_to_layout()                 # patches layout.walkable in-place
"""
from __future__ import annotations

from collections import deque
from typing import Set, Tuple

import numpy as np

from src.environment.building import BuildingLayout


class SmokeLayer:
    """Cellular-automaton smoke diffusion on the building grid.

    Parameters
    ----------
    layout:
        The :class:`BuildingLayout` instance. Smoke only spreads through
        walkable cells (it can't pass through solid walls).
    origin:
        Fire origin in world coordinates (x_m, y_m).
    spread_rate:
        How many meters of smoke radius are added per second.
        E.g. 0.4 m/s means smoke reaches 8 m after 20 s.
    visibility_threshold:
        Smoke density [0, 1] above which a cell is considered impassable.
        Defaults to 0.5.
    max_density:
        Maximum smoke density any cell can reach (cap at 1.0).
    """

    def __init__(
        self,
        layout: BuildingLayout,
        origin: tuple[float, float],
        spread_rate: float = 0.4,
        visibility_threshold: float = 0.5,
        max_density: float = 1.0,
    ) -> None:
        self.layout = layout
        self.origin_world = np.array(origin, dtype=float)
        self.spread_rate = spread_rate
        self.visibility_threshold = visibility_threshold
        self.max_density = max_density

        # Smoke density grid [0, max_density] — one value per cell.
        self.density = np.zeros(
            (layout.grid_height, layout.grid_width), dtype=float
        )
        # Seed the origin cell with full density.
        self._origin_cell = layout.world_to_cell(self.origin_world)
        if layout.is_walkable_cell(self._origin_cell):
            ox, oy = self._origin_cell
            self.density[oy, ox] = max_density

        self._current_radius: float = 0.0
        # Cache the original walkable state to allow full reset if needed.
        self._original_walkable: np.ndarray = layout.walkable.copy()

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    def step(self, current_time: float, dt: float) -> None:
        """Advance smoke one time-step.

        Smoke spreads outward from the origin via BFS up to the current
        cumulative radius.  Cells are filled proportionally to distance
        so there's a soft density gradient.
        """
        self._current_radius = self.spread_rate * current_time
        self._recompute_density()

    def compute_visual_density(self, current_time: float) -> np.ndarray:
        """Return a density grid for *rendering only* at the given time.

        Uses the **original** (pre-smoke) walkable map so the orange overlay
        grows naturally over the floor plan even though the navigation grid
        has already been patched with the worst-case smoke state.
        The model state (``self.density``) is not modified.
        """
        radius = self.spread_rate * current_time
        visual = np.zeros((self.layout.grid_height, self.layout.grid_width), dtype=float)
        if radius <= 0.0:
            return visual

        ox, oy = self._origin_cell
        orig = self._original_walkable
        w, h = self.layout.grid_width, self.layout.grid_height
        cell_size = self.layout.cell_size

        visited: Set[Tuple[int, int]] = set()
        queue: deque[tuple[int, int, float]] = deque([(ox, oy, 0.0)])
        visited.add((ox, oy))

        while queue:
            cx, cy, dist = queue.popleft()
            if dist > radius:
                continue
            frac = max(0.0, 1.0 - dist / radius)
            visual[cy, cx] = self.max_density * frac

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not orig[ny, nx]:
                    continue
                step_cost = cell_size * (np.sqrt(2) if dx and dy else 1.0)
                new_dist = dist + step_cost
                if new_dist <= radius:
                    visited.add((nx, ny))
                    queue.append((nx, ny, new_dist))

        return visual

    def _recompute_density(self) -> None:
        """BFS from origin up to current radius, setting density by distance.

        Uses ``_original_walkable`` so that smoke-blocked cells (already
        marked non-walkable in ``layout.walkable``) do not prevent further
        smoke spread — smoke travels through rooms regardless of current
        blocking state.
        """
        new_density = np.zeros_like(self.density)
        ox, oy = self._origin_cell
        orig = self._original_walkable
        # Guard: origin must have been walkable in the original layout.
        if not orig[oy, ox]:
            return

        w, h = self.layout.grid_width, self.layout.grid_height
        # BFS with distance tracking.
        visited: Set[Tuple[int, int]] = set()
        queue: deque[tuple[int, int, float]] = deque()
        queue.append((ox, oy, 0.0))
        visited.add((ox, oy))
        cell_size = self.layout.cell_size

        while queue:
            cx, cy, dist = queue.popleft()
            if dist > self._current_radius:
                continue

            # Density falls off linearly with distance from origin.
            frac = max(0.0, 1.0 - dist / max(self._current_radius, 0.01))
            new_density[cy, cx] = self.max_density * frac

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                step_cost = cell_size * (np.sqrt(2) if dx != 0 and dy != 0 else 1.0)
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if not orig[ny, nx]:
                    continue
                new_dist = dist + step_cost
                if new_dist <= self._current_radius:
                    visited.add((nx, ny))
                    queue.append((nx, ny, new_dist))

        self.density = new_density

    def blocked_cells(self) -> Set[Tuple[int, int]]:
        """Return the set of grid cells where smoke exceeds the threshold."""
        ys, xs = np.where(self.density >= self.visibility_threshold)
        return set(zip(xs.tolist(), ys.tolist()))

    def apply_to_layout(self) -> None:
        """Patch `layout.walkable` in-place: mark smoke-filled cells as blocked.

        Call this before recomputing the floor field each scenario step (or
        once at the start if smoke is treated as static).
        """
        # Restore original walkable first, then block smoke cells.
        self.layout.walkable[:] = self._original_walkable
        # Keep exit cells open — smoke should not block the exit cell itself
        # (agents escape *through* the opening).
        blocked_mask = self.density >= self.visibility_threshold
        # Do not block cells that are exit cells.
        exit_cell_mask = np.zeros_like(blocked_mask, dtype=bool)
        for exit_def in self.layout.exits.values():
            for xc, yc in exit_def.cells:
                if 0 <= yc < self.layout.grid_height and 0 <= xc < self.layout.grid_width:
                    exit_cell_mask[yc, xc] = True

        apply_mask = blocked_mask & ~exit_cell_mask
        self.layout.walkable[apply_mask] = False
        self.layout._blocked_centers_cache = None

    def restore_layout(self) -> None:
        """Undo all smoke patches from layout.walkable."""
        self.layout.walkable[:] = self._original_walkable
        self.layout._blocked_centers_cache = None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exits_blocked_by_smoke(self, threshold: float | None = None) -> list[str]:
        """Return names of exits where the approach corridor is smoke-filled.

        An exit is considered blocked when a majority of cells within
        ``approach_depth`` meters of the exit center exceed the threshold.
        """
        if threshold is None:
            threshold = self.visibility_threshold

        blocked: list[str] = []
        approach_depth = self.layout.cell_size * 3  # look 3 cells deep
        for name, exit_def in self.layout.exits.items():
            center = exit_def.center
            # Check 5 cells inward from the exit along its axis.
            side = exit_def.side
            offsets = {
                "west":  np.array([1.0, 0.0]),
                "east":  np.array([-1.0, 0.0]),
                "north": np.array([0.0, -1.0]),
                "south": np.array([0.0, 1.0]),
            }[side]
            smoked_count = 0
            for depth in range(1, 6):
                probe = center + offsets * depth * self.layout.cell_size
                cx, cy = self.layout.world_to_cell(probe)
                if 0 <= cy < self.layout.grid_height and 0 <= cx < self.layout.grid_width:
                    if self.density[cy, cx] >= threshold:
                        smoked_count += 1
            if smoked_count >= 3:
                blocked.append(name)
        return blocked
