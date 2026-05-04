from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.models.social_force import SocialForceParameters


@dataclass(slots=True)
class SimulationParameters:
    """Top-level parameters required by the baseline simulation."""

    dt: float
    max_time: float
    seed: int
    record_interval: int


@dataclass(slots=True)
class AgentParameters:
    """Randomized baseline parameters for regular evacuees."""

    total: int
    desired_speed_mean: float
    desired_speed_std: float
    mass_mean: float
    mass_std: float
    radius_min: float
    radius_max: float
    reaction_time_scale: float
    reaction_time_sigma: float
    relaxation_time: float
    max_speed_factor: float


@dataclass(slots=True)
class VisualizationParameters:
    """Controls for the matplotlib animation."""

    interval_ms: int
    trail_length: int


@dataclass(slots=True)
class BaselineScenario:
    """Fully parsed scenario used by the simulation model."""

    simulation: SimulationParameters
    agents: AgentParameters
    social_force: SocialForceParameters
    building: dict[str, float]
    visualization: VisualizationParameters


def load_baseline_scenario(config_path: str | Path) -> BaselineScenario:
    """Load baseline settings from YAML."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config: dict[str, Any] = yaml.safe_load(handle)

    return BaselineScenario(
        simulation=SimulationParameters(**raw_config["simulation"]),
        agents=AgentParameters(**raw_config["agents"]),
        social_force=SocialForceParameters(**raw_config["social_force"]),
        building=raw_config["building"],
        visualization=VisualizationParameters(**raw_config["visualization"]),
    )
