from __future__ import annotations

import numpy as np

from src.agents.protocols import AgentLike
from src.environment.floor_field import StaticFloorField


class NavigationModel:
    """Simple exit assignment and floor-field guidance."""

    def __init__(self, floor_field: StaticFloorField) -> None:
        self.floor_field = floor_field

    def assign_exit(self, agent: AgentLike) -> None:
        """Assign the nearest reachable exit once at initialization time."""
        if agent.exit_name is None:
            agent.exit_name = self.floor_field.best_exit(agent.position)

    def best_exit(self, point: np.ndarray, excluded_exits: set[str] | None = None) -> str:
        """Return the available exit with the smallest geodesic distance.

        Parameters
        ----------
        point:
            Agent world position.
        excluded_exits:
            Exit names to ignore (e.g. smoke-blocked or disabled exits).
        """
        x_index, y_index = self.floor_field.layout.world_to_cell(point)
        available = {
            name: dmap
            for name, dmap in self.floor_field.distance_maps.items()
            if excluded_exits is None or name not in excluded_exits
        }
        if not available:
            # Fallback: use all maps if every exit is somehow excluded.
            available = self.floor_field.distance_maps
        return min(available, key=lambda name: available[name][y_index, x_index])

    def desired_direction(self, agent: AgentLike) -> np.ndarray:
        """Return the movement direction given the agent's assigned exit."""

        if agent.exit_name is None:
            raise ValueError("Agent must have an assigned exit before movement starts.")
        return self.floor_field.direction_for_exit(agent.position, agent.exit_name)
