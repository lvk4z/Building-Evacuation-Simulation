"""Mesa-based evacuation model — core simulation loop.

This module wires together the Social Force Model physics, the building
layout, the static floor field navigation, and Mesa's infrastructure
(ContinuousSpace, DataCollector, AgentSet) into a single
:class:`EvacuationModel`.

Usage example::

    from src.simulation.mesa_model import EvacuationModel
    from src.simulation.scenario import load_baseline_scenario

    scenario = load_baseline_scenario("config.yaml")
    model = EvacuationModel(scenario)
    result = model.run_simulation()
"""
from __future__ import annotations

from dataclasses import dataclass

import mesa
import numpy as np
from mesa import DataCollector
from mesa.space import ContinuousSpace
from scipy.spatial import cKDTree

from src.agents.evacuee_agent import EvacueeAgent
from src.analysis.metrics import SimulationMetrics, compute_metrics
from src.environment.building import BuildingLayout
from src.environment.floor_field import StaticFloorField
from src.models.navigation import NavigationModel
from src.models.social_force import (
    SocialForceParameters,
    agent_repulsion,
    driving_force,
    limit_speed,
    obstacle_repulsion,
)
from src.simulation.scenario import AgentParameters, BaselineScenario


@dataclass(slots=True)
class FrameSnapshot:
    """Lightweight snapshot of all agent positions at one time step."""

    time: float
    positions: np.ndarray
    speeds: np.ndarray
    active_mask: np.ndarray
    exited_mask: np.ndarray
    exit_names: np.ndarray


@dataclass
class SimulationResult:
    """Return value of :meth:`EvacuationModel.run_simulation`."""

    model: "EvacuationModel"
    frames: list[FrameSnapshot]
    metrics: SimulationMetrics
    dt: float

    @property
    def layout(self) -> BuildingLayout:
        return self.model.layout

    @property
    def floor_field(self) -> StaticFloorField:
        return self.model.floor_field

    @property
    def agents(self) -> list[EvacueeAgent]:
        return list(self.model.agents)


class EvacuationModel(mesa.Model):
    """Mesa Model driving the baseline building evacuation scenario.

    The model owns all environment objects (building layout, floor field,
    navigation, SFM parameters) and runs the per-step physics loop:

    1.  Collect position / velocity vectors from *active* agents.
    2.  Compute vectorised SFM forces (driving, social, obstacle).
    3.  Integrate velocity and resolve wall collisions.
    4.  Commit updates via :meth:`EvacueeAgent.apply_update`.
    5.  Detect and mark agents that reach their exit.
    6.  Record a :class:`FrameSnapshot` at the configured cadence.
    7.  Ask :class:`~mesa.DataCollector` to collect model / agent reporters.

    Parameters
    ----------
    scenario:
        Fully configured :class:`~src.simulation.scenario.BaselineScenario`.
    """

    def __init__(self, scenario: BaselineScenario) -> None:
        # Mesa Model uses `rng` to seed numpy's Generator.
        super().__init__(rng=scenario.simulation.seed)

        # ── Environment ───────────────────────────────────────────────
        self.layout = BuildingLayout.baseline(**scenario.building)
        self.floor_field = StaticFloorField(self.layout)
        self.navigation = NavigationModel(self.floor_field)
        self.sfm_params: SocialForceParameters = scenario.social_force

        self.obstacle_points = self.layout.get_blocked_cell_centers()
        self.obstacle_tree = (
            cKDTree(self.obstacle_points) if self.obstacle_points.size else None
        )

        # ── Simulation parameters ─────────────────────────────────────
        self.dt: float = scenario.simulation.dt
        self.max_time: float = scenario.simulation.max_time
        self.record_interval: int = max(1, scenario.simulation.record_interval)
        self.current_time: float = 0.0
        self._step_count: int = 0

        # ── Mesa ContinuousSpace ──────────────────────────────────────
        # Agents live in a 2-D space matching the building dimensions.
        self.space = ContinuousSpace(
            x_max=self.layout.width_m,
            y_max=self.layout.height_m,
            torus=False,
        )

        # ── Create agents ─────────────────────────────────────────────
        self._create_agents(scenario.agents)

        # ── Mesa DataCollector ────────────────────────────────────────
        self.datacollector = DataCollector(
            model_reporters={
                "time": "current_time",
                "evacuated": "evacuated_count",
                "remaining": "remaining_count",
                "mean_speed": "mean_active_speed",
            },
            agent_reporters={
                "x": lambda a: float(a.position[0]),
                "y": lambda a: float(a.position[1]),
                "speed": lambda a: float(np.linalg.norm(a.velocity)),
                "exited": "exited",
                "exit_name": "exit_name",
                "agent_type": "agent_type",
                "reaction_time": "reaction_time",
            },
        )

        # ── Frame buffer for animation ────────────────────────────────
        self.frames: list[FrameSnapshot] = []
        self._record_frame()

        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _sample_reaction_time(self, params: AgentParameters) -> float:
        return float(
            self.rng.lognormal(
                mean=np.log(params.reaction_time_scale),
                sigma=params.reaction_time_sigma,
            )
        )

    def _create_agents(self, params: AgentParameters) -> None:
        positions = self.layout.sample_spawn_positions(
            count=params.total, rng=self.rng, min_distance=0.62
        )

        for position in positions:
            desired_speed = float(
                max(0.6, self.rng.normal(params.desired_speed_mean, params.desired_speed_std))
            )
            mass = float(max(45.0, self.rng.normal(params.mass_mean, params.mass_std)))
            radius = float(self.rng.uniform(params.radius_min, params.radius_max))
            reaction_time = self._sample_reaction_time(params)

            agent = EvacueeAgent(
                model=self,
                position=position,
                desired_speed=desired_speed,
                max_speed=desired_speed * params.max_speed_factor,
                mass=mass,
                radius=radius,
                relaxation_time=params.relaxation_time,
                reaction_time=reaction_time,
                agent_type="regular",
            )
            # Assign the nearest exit using the static floor field.
            self.navigation.assign_exit(agent)
            # Register position in the Mesa ContinuousSpace.
            self.space.place_agent(agent, (float(position[0]), float(position[1])))

    # ------------------------------------------------------------------
    # Mesa Model.step() — called once per dt tick
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one time step (dt seconds).

        The method follows a simultaneous-activation pattern:
        all forces are evaluated using positions from the *current* step
        before any position is updated.  Updates are applied atomically
        at the end of the step.
        """
        if self.all_evacuated or self.current_time >= self.max_time:
            self.running = False
            return

        agents_list: list[EvacueeAgent] = list(self.agents)
        n = len(agents_list)

        positions = np.array([a.position for a in agents_list], dtype=float)
        velocities = np.array([a.velocity for a in agents_list], dtype=float)
        active_mask = np.array(
            [a.is_active(self.current_time) for a in agents_list], dtype=bool
        )
        active_indices = np.flatnonzero(active_mask)

        # Build neighbour tree for active (non-exited, post-reaction) agents.
        active_tree = (
            cKDTree(positions[active_mask]) if active_indices.size else None
        )

        next_positions = positions.copy()
        next_velocities = velocities.copy()

        for agent_index in active_indices:
            agent = agents_list[agent_index]

            # Dynamic exit selection: pick closest at each step.
            agent.exit_name = self.floor_field.best_exit(agent.position)
            desired_direction = self.navigation.desired_direction(agent)

            # --- SFM force accumulation ---
            total_force = driving_force(agent, desired_direction)

            if active_tree is not None:
                neighbour_local_indices = active_tree.query_ball_point(
                    positions[agent_index], r=self.sfm_params.neighbour_radius
                )
                neighbour_indices = [
                    int(active_indices[idx])
                    for idx in neighbour_local_indices
                    if int(active_indices[idx]) != agent_index
                ]
            else:
                neighbour_indices = []

            total_force += agent_repulsion(
                agent_index=agent_index,
                agents=agents_list,
                positions=positions,
                velocities=velocities,
                neighbour_indices=neighbour_indices,
                params=self.sfm_params,
            )
            total_force += obstacle_repulsion(
                agent=agent,
                layout=self.layout,
                obstacle_tree=self.obstacle_tree,
                obstacle_points=self.obstacle_points,
                params=self.sfm_params,
            )

            # --- Euler integration ---
            acceleration = total_force / max(agent.mass, 1e-6)
            proposed_velocity = velocities[agent_index] + acceleration * self.dt
            proposed_velocity = limit_speed(proposed_velocity, agent.max_speed)
            proposed_position = positions[agent_index] + proposed_velocity * self.dt
            resolved_position = self.layout.resolve_move(
                positions[agent_index], proposed_position
            )
            move_distance = float(
                np.linalg.norm(resolved_position - positions[agent_index])
            )

            if move_distance < self.layout.cell_size * 0.08:
                # Fallback nudge when force model stalls near obstacles.
                resolved_position = self._nudge_toward_better_cell(
                    agent, positions[agent_index]
                )
                move_distance = float(
                    np.linalg.norm(resolved_position - positions[agent_index])
                )

            if move_distance < 1e-8:
                proposed_velocity *= 0.2

            next_positions[agent_index] = resolved_position
            next_velocities[agent_index] = proposed_velocity

        # --- Commit updates (simultaneous activation) ---
        self.current_time += self.dt
        self._step_count += 1

        for agent_index, agent in enumerate(agents_list):
            if agent.exited:
                continue

            agent.apply_update(next_positions[agent_index], next_velocities[agent_index])
            # Keep Mesa ContinuousSpace in sync.
            self.space.move_agent(
                agent,
                (float(agent.position[0]), float(agent.position[1])),
            )

            if agent.exit_name is not None and self.layout.is_at_exit(
                agent.position, agent.exit_name
            ):
                agent.mark_exited(self.current_time)

        # --- Record frame for animation ---
        if self._step_count % self.record_interval == 0:
            self._record_frame()

        # --- Collect Mesa metrics ---
        self.datacollector.collect(self)

        if self.all_evacuated:
            self.running = False

    # ------------------------------------------------------------------
    def run_simulation(self) -> SimulationResult:
        """Run until all agents evacuate or max_time is reached.

        Returns a :class:`SimulationResult` with frames for animation and
        summary metrics.
        """
        self.running = True

        while self.running:
            self.step()

        # Ensure the final state is captured in the frame buffer.
        if not self.frames or self.frames[-1].time < self.current_time:
            self._record_frame()

        agents_list: list[EvacueeAgent] = list(self.agents)
        metrics = compute_metrics(agents_list, self.current_time)

        return SimulationResult(
            model=self,
            frames=self.frames,
            metrics=metrics,
            dt=self.dt,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_evacuated(self) -> bool:
        """Return True when every agent has left the building."""
        return all(a.exited for a in self.agents)

    @property
    def evacuated_count(self) -> int:
        """Number of agents that have already exited."""
        return sum(1 for a in self.agents if a.exited)

    @property
    def remaining_count(self) -> int:
        """Number of agents still inside the building."""
        return sum(1 for a in self.agents if not a.exited)

    @property
    def mean_active_speed(self) -> float:
        """Mean speed of agents that have not exited yet."""
        active_speeds = [np.linalg.norm(a.velocity) for a in self.agents if not a.exited]
        return float(np.mean(active_speeds)) if active_speeds else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nudge_toward_better_cell(
        self, agent: EvacueeAgent, old_point: np.ndarray
    ) -> np.ndarray:
        """Short deterministic step toward the best floor-field neighbour.

        Prevents stalling on the coarse grid near door frames and columns.
        """
        if agent.exit_name is None:
            return old_point

        current_cell = self.layout.world_to_cell(old_point)
        distance_map = self.floor_field.distance_maps[agent.exit_name]
        best_cell = None
        best_distance = float("inf")

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                candidate = (current_cell[0] + dx, current_cell[1] + dy)
                if not self.layout.is_walkable_cell(candidate):
                    continue
                candidate_distance = float(
                    distance_map[candidate[1], candidate[0]]
                )
                if candidate_distance < best_distance:
                    best_distance = candidate_distance
                    best_cell = candidate

        if best_cell is None:
            return old_point

        target = self.layout.cell_to_world(best_cell)
        direction = target - old_point
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return old_point

        nudged = old_point + direction / norm * min(norm, self.layout.cell_size * 0.45)
        return self.layout.resolve_move(old_point, nudged)

    def _record_frame(self) -> None:
        """Append the current agent states to the frame buffer."""
        agents_list: list[EvacueeAgent] = list(self.agents)

        self.frames.append(
            FrameSnapshot(
                time=self.current_time,
                positions=np.array([a.position for a in agents_list], dtype=float),
                speeds=np.array(
                    [float(np.linalg.norm(a.velocity)) for a in agents_list],
                    dtype=float,
                ),
                active_mask=np.array(
                    [a.is_active(self.current_time) for a in agents_list], dtype=bool
                ),
                exited_mask=np.array([a.exited for a in agents_list], dtype=bool),
                exit_names=np.array(
                    [a.exit_name or "" for a in agents_list], dtype="U8"
                ),
            )
        )
