from __future__ import annotations

from typing import Protocol

import numpy as np


class AgentLike(Protocol):
    """Minimal agent interface shared by physics, navigation, and metrics."""

    position: np.ndarray
    velocity: np.ndarray
    desired_speed: float
    mass: float
    radius: float
    relaxation_time: float
    exit_name: str | None
    exited: bool
    exit_time: float | None

    def is_active(self, current_time: float) -> bool:
        """Return whether the agent is already participating in evacuation."""
