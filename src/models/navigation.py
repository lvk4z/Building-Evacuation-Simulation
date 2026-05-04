from __future__ import annotations

import numpy as np

from src.agents.protocols import AgentLike
from src.environment.floor_field import StaticFloorField


class NavigationModel:
    """Simple exit assignment and floor-field guidance."""

    def __init__(self, floor_field: StaticFloorField) -> None:
        self.floor_field = floor_field

    def assign_exit(self, agent: AgentLike) -> None:
        """Assign the nearest exit once at initialization time."""

        if agent.exit_name is None:
            agent.exit_name = self.floor_field.best_exit(agent.position)

    def desired_direction(self, agent: AgentLike) -> np.ndarray:
        """Return the movement direction given the agent's assigned exit."""

        if agent.exit_name is None:
            raise ValueError("Agent must have an assigned exit before movement starts.")
        return self.floor_field.direction_for_exit(agent.position, agent.exit_name)
