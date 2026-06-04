from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.social_force import SocialForceParameters


class ScenarioType(str, Enum):
    BASELINE = "baseline"
    SMOKE = "smoke"
    LEADERS = "leaders"


@dataclass
class SmokeConfig:
    """Parameters for Scenario B — fire + smoke."""

    fire_origin_x: float       # world x [m] — fire source
    fire_origin_y: float       # world y [m]
    spread_rate: float         # smoke radius growth [m/s]
    visibility_threshold: float  # density above which cell is blocked [0-1]
    # Exits to unconditionally disable (in addition to dynamic smoke blocking)
    disabled_exits: list[str] = field(default_factory=list)


@dataclass
class LeaderConfig:
    """Parameters for Scenario C — leader agents."""

    fraction: float            # fraction of agents that are leaders [0-1]
    speed_mean: float          # leader desired speed [m/s]
    speed_std: float           # speed standard deviation
    reaction_time_scale: float  # median reaction time [s] (faster than regular)


@dataclass
class SimulationParameters:
    """Top-level parameters required by the baseline simulation."""

    dt: float
    max_time: float
    seed: int
    record_interval: int


@dataclass
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


@dataclass
class VisualizationParameters:
    """Controls for the matplotlib animation."""

    interval_ms: int
    trail_length: int


@dataclass
class BaselineScenario:
    """Fully parsed scenario used by the simulation model."""

    scenario_type: ScenarioType
    simulation: SimulationParameters
    agents: AgentParameters
    social_force: SocialForceParameters
    building: dict[str, float]
    visualization: VisualizationParameters
    smoke: Optional[SmokeConfig] = None
    leaders: Optional[LeaderConfig] = None


def load_baseline_scenario(config_path: str | Path) -> BaselineScenario:
    """Load baseline settings from YAML."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw_config: dict[str, Any] = yaml.safe_load(handle)

    return BaselineScenario(
        scenario_type=ScenarioType.BASELINE,
        simulation=SimulationParameters(**raw_config["simulation"]),
        agents=AgentParameters(**raw_config["agents"]),
        social_force=SocialForceParameters(**raw_config["social_force"]),
        building=raw_config["building"],
        visualization=VisualizationParameters(**raw_config["visualization"]),
        smoke=None,
        leaders=None,
    )


def load_smoke_scenario(config_path: str | Path) -> BaselineScenario:
    """Load Scenario B — fire in conference room, smoke blocking east corridor."""

    scenario = load_baseline_scenario(config_path)
    # Fire starts in the bottom-center room (conference room, approx center)
    smoke_cfg = SmokeConfig(
        fire_origin_x=18.0,
        fire_origin_y=4.0,
        spread_rate=0.4,
        visibility_threshold=0.5,
        disabled_exits=["E2"],  # east exit blocked structurally
    )
    object.__setattr__(scenario, "scenario_type", ScenarioType.SMOKE)
    object.__setattr__(scenario, "smoke", smoke_cfg)
    return scenario


def load_leaders_scenario(config_path: str | Path, leader_fraction: float = 0.10) -> BaselineScenario:
    """Load Scenario C — population includes leader agents."""

    scenario = load_baseline_scenario(config_path)
    leader_cfg = LeaderConfig(
        fraction=leader_fraction,
        speed_mean=1.0,
        speed_std=0.1,
        reaction_time_scale=5.0,
    )
    object.__setattr__(scenario, "scenario_type", ScenarioType.LEADERS)
    object.__setattr__(scenario, "leaders", leader_cfg)
    return scenario
