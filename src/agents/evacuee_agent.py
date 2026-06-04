"""Mesa-based evacuee agent for the building evacuation simulation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import mesa
import numpy as np

if TYPE_CHECKING:
    from src.simulation.mesa_model import EvacuationModel


class EvacueeAgent(mesa.Agent):
    """A single evacuee modelled as a Mesa Agent.

    Physical movement is computed in bulk by :class:`EvacuationModel` using
    the vectorised Social Force Model.  The agent's ``step()`` method is
    intentionally empty: the model accumulates forces for all active agents
    simultaneously and then calls :meth:`apply_update` to commit the result.

    Attributes
    ----------
    position:    numpy array [m] — world coordinates used by the SFM.
    velocity:    numpy array [m/s].
    desired_speed, max_speed, mass, radius, relaxation_time:  SFM body params.
    reaction_time: pre-evacuation delay [s] (LogNormal distributed).
    exit_name:   assigned exit label; updated each simulation step.
    exited:      True once the agent passes its exit opening.
    exit_time:   simulation time [s] when the agent exited, or None.
    agent_type:  string label used for statistics / visualisation.
    """

    def __init__(
        self,
        model: EvacuationModel,
        position: np.ndarray,
        desired_speed: float,
        max_speed: float,
        mass: float,
        radius: float,
        relaxation_time: float,
        reaction_time: float,
        agent_type: str = "regular",
        preferred_exit: str | None = None,
    ) -> None:
        super().__init__(model)

        self.position: np.ndarray = position.copy()
        self.velocity: np.ndarray = np.zeros(2, dtype=float)

        self.desired_speed: float = desired_speed
        self.max_speed: float = max_speed
        self.mass: float = mass
        self.radius: float = radius
        self.relaxation_time: float = relaxation_time

        self.reaction_time: float = reaction_time
        self.exit_name: Optional[str] = None
        self.exited: bool = False
        self.exit_time: Optional[float] = None
        self.agent_type: str = agent_type
        # Leaders use a fixed preferred exit; None means dynamic best-exit.
        self.preferred_exit: Optional[str] = preferred_exit

        self.pos: Optional[tuple[float, float]] = None

    def step(self) -> None:
        """No-op — physics are computed centrally in EvacuationModel.step()."""

    def is_active(self, current_time: float) -> bool:
        """Return True once pre-evacuation delay has elapsed."""
        return (not self.exited) and current_time >= self.reaction_time

    def apply_update(self, new_position: np.ndarray, new_velocity: np.ndarray) -> None:
        """Commit a position/velocity update computed by the model."""
        self.position = new_position
        self.velocity = new_velocity

    def mark_exited(self, current_time: float) -> None:
        """Freeze the agent once it leaves through its assigned exit."""
        self.exited = True
        self.exit_time = current_time
        self.velocity = np.zeros(2, dtype=float)

    def __repr__(self) -> str:
        return (
            f"EvacueeAgent(id={self.unique_id}, type={self.agent_type}, "
            f"exited={self.exited}, pos={self.position})"
        )
