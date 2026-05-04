from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.agents.protocols import AgentLike


@dataclass(slots=True)
class SimulationMetrics:
    """Compact summary for a single simulation run."""

    total_evacuation_time: float
    evacuated_agents: int
    total_agents: int
    mean_exit_time: float
    median_exit_time: float
    completed: bool


def compute_metrics(agents: list[AgentLike], current_time: float) -> SimulationMetrics:
    """Aggregate run-level metrics for a single simulation run."""

    exit_times = np.array([agent.exit_time for agent in agents if agent.exit_time is not None], dtype=float)
    evacuated_agents = int(exit_times.size)
    total_agents = len(agents)
    completed = evacuated_agents == total_agents

    if evacuated_agents == 0:
        return SimulationMetrics(
            total_evacuation_time=current_time,
            evacuated_agents=0,
            total_agents=total_agents,
            mean_exit_time=float("nan"),
            median_exit_time=float("nan"),
            completed=False,
        )

    return SimulationMetrics(
        total_evacuation_time=float(np.max(exit_times) if completed else current_time),
        evacuated_agents=evacuated_agents,
        total_agents=total_agents,
        mean_exit_time=float(np.mean(exit_times)),
        median_exit_time=float(np.median(exit_times)),
        completed=completed,
    )
