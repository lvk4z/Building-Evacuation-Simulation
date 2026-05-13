from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.agents.protocols import AgentLike
from src.environment.building import BuildingLayout


@dataclass(slots=True)
class SocialForceParameters:
    """Core parameters of the simplified social-force model."""

    A: float
    B: float
    k: float
    kappa: float
    obstacle_strength: float
    obstacle_range: float
    neighbour_radius: float


def driving_force(agent: AgentLike, desired_direction: np.ndarray) -> np.ndarray:
    """Acceleration term pulling the agent toward its desired velocity."""

    desired_velocity = agent.desired_speed * desired_direction
    return agent.mass * (desired_velocity - agent.velocity) / max(agent.relaxation_time, 1e-3)


def agent_repulsion(
    agent_index: int,
    agents: list[AgentLike],
    positions: np.ndarray,
    velocities: np.ndarray,
    neighbour_indices: list[int],
    params: SocialForceParameters,
) -> np.ndarray:
    """Pairwise social force between nearby agents.

    The expression follows the classic Helbing-style formulation, with the
    contact term activated only when bodies overlap.
    """

    agent = agents[agent_index]
    total_force = np.zeros(2, dtype=float)

    for other_index in neighbour_indices:
        if other_index == agent_index:
            continue

        other = agents[other_index]
        delta = positions[agent_index] - positions[other_index]
        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            delta = np.array([1e-4, 0.0])
            distance = np.linalg.norm(delta)

        normal = delta / distance
        tangent = np.array([-normal[1], normal[0]])
        overlap = agent.radius + other.radius - distance

        force = params.A * np.exp(overlap / max(params.B, 1e-6)) * normal
        if overlap > 0.0:
            tangential_relative_velocity = np.dot(velocities[other_index] - velocities[agent_index], tangent)
            force += params.k * overlap * normal
            force += -params.kappa * overlap * tangential_relative_velocity * tangent

        total_force += force

    return total_force


def obstacle_repulsion(
    agent: AgentLike,
    layout: BuildingLayout,
    obstacle_tree: cKDTree | None,
    obstacle_points: np.ndarray,
    params: SocialForceParameters,
) -> np.ndarray:
    """Approximate wall repulsion using blocked grid-cell centers."""

    if obstacle_tree is None or obstacle_points.size == 0:
        return np.zeros(2, dtype=float)

    indices = obstacle_tree.query_ball_point(agent.position, r=params.neighbour_radius)
    if not indices:
        return np.zeros(2, dtype=float)

    total_force = np.zeros(2, dtype=float)
    padding = layout.cell_size * 0.35

    for obstacle_index in indices:
        obstacle_point = obstacle_points[obstacle_index]
        delta = agent.position - obstacle_point
        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            continue

        surface_distance = max(distance - padding, 1e-6)
        normal = delta / distance
        overlap = agent.radius - surface_distance
        magnitude = params.obstacle_strength * np.exp(overlap / max(params.obstacle_range, 1e-6))
        if overlap > 0.0:
            magnitude += params.k * overlap
        total_force += magnitude * normal

    return total_force


def limit_speed(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    """Clamp agent speed for numerical stability."""

    speed = np.linalg.norm(velocity)
    if speed <= max_speed:
        return velocity
    return velocity * (max_speed / max(speed, 1e-8))
