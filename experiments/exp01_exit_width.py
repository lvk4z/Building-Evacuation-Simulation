"""Experiment 1 — Effect of exit E3 width on T_evac and flow rate.

Vary E3 (north exit) width: 0.8 / 1.0 / 1.2 / 1.5 / 2.0 m.
All other parameters are held at baseline.  190 agents, 10 MC seeds.

Run:
    python experiments/exp01_exit_width.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from mesa import DataCollector
from mesa.space import ContinuousSpace
from scipy.spatial import cKDTree

from src.environment.building import BuildingLayout, SpawnZone
from src.environment.floor_field import StaticFloorField
from src.environment.smoke_model import SmokeLayer
from src.models.navigation import NavigationModel
from src.simulation.mesa_model import EvacuationModel
from src.simulation.scenario import load_baseline_scenario
from experiments._runner import save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
SEEDS = list(range(10))
E3_WIDTHS = [0.8, 1.0, 1.2, 1.5, 2.0]


def build_layout_custom_e3(e3_width: float) -> BuildingLayout:
    layout = BuildingLayout(width_m=36.0, height_m=20.0, cell_size=0.4)
    layout.walkable[0, :] = False; layout.walkable[-1, :] = False
    layout.walkable[:, 0] = False; layout.walkable[:, -1] = False
    layout._add_horizontal_wall(y=8.0, door_spans=[(5.1, 6.9), (17.1, 18.9), (29.1, 30.9)])
    layout._add_horizontal_wall(y=12.0, door_spans=[(3.6, 5.4), (12.6, 14.4), (21.6, 23.4), (30.6, 32.4)])
    for x in (9.0, 18.0, 27.0):
        layout._add_vertical_wall(x=x, y_min=12.0, y_max=20.0, door_spans=[])
    for x in (12.0, 24.0):
        layout._add_vertical_wall(x=x, y_min=0.0, y_max=8.0, door_spans=[])
    layout._fill_rectangle(13.6, 15.0, 8.9, 11.1, walkable=False)
    layout._fill_rectangle(21.2, 22.6, 8.9, 11.1, walkable=False)
    layout._register_exit("E1", "west",  10.0, 2.0)
    layout._register_exit("E2", "east",  10.0, 1.2)
    layout._register_exit("E3", "north", 22.5, e3_width)
    layout._register_exit("E4", "south", 18.0, 1.8)
    layout.spawn_zones = [
        SpawnZone("top_left_1",    1.0,  8.0, 13.0, 19.0),
        SpawnZone("top_left_2",   10.0, 17.0, 13.0, 19.0),
        SpawnZone("top_right_1",  19.0, 26.0, 13.0, 19.0),
        SpawnZone("top_right_2",  28.0, 35.0, 13.0, 19.0),
        SpawnZone("bottom_left",   1.0, 11.0,  1.0,  7.0),
        SpawnZone("bottom_center",13.0, 23.0,  1.0,  7.0),
        SpawnZone("bottom_right", 25.0, 35.0,  1.0,  7.0),
    ]
    return layout


class _CustomLayoutModel(EvacuationModel):
    """EvacuationModel with a pre-built layout (bypasses baseline() factory)."""

    def __init__(self, scenario, custom_layout: BuildingLayout) -> None:
        import mesa
        mesa.Model.__init__(self, rng=scenario.simulation.seed)

        self.layout = custom_layout
        self.scenario_type = scenario.scenario_type
        self.smoke: SmokeLayer | None = None
        self._disabled_exits: set[str] = set()

        self.floor_field = StaticFloorField(self.layout)
        self.navigation = NavigationModel(self.floor_field)
        self.sfm_params = scenario.social_force

        pts = self.layout.get_blocked_cell_centers()
        self.obstacle_points = pts
        self.obstacle_tree = cKDTree(pts) if pts.size else None

        self.dt = scenario.simulation.dt
        self.max_time = scenario.simulation.max_time
        self.record_interval = max(1, scenario.simulation.record_interval)
        self.current_time = 0.0
        self._step_count = 0

        self.space = ContinuousSpace(self.layout.width_m, self.layout.height_m, False)
        self._create_agents(scenario)

        self.datacollector = DataCollector(
            model_reporters={"time": "current_time", "evacuated": "evacuated_count",
                             "remaining": "remaining_count", "mean_speed": "mean_active_speed"},
            agent_reporters={"speed": lambda a: float(np.linalg.norm(a.velocity)),
                             "exited": "exited", "agent_type": "agent_type"},
        )
        self.frames = []
        self._record_frame()
        self.datacollector.collect(self)


if __name__ == "__main__":
    all_rows = []
    for width in E3_WIDTHS:
        print(f"\n--- E3 width = {width} m ---")
        for seed in SEEDS:
            sc = load_baseline_scenario(CONFIG)
            sc.simulation.seed = seed
            lay = build_layout_custom_e3(width)
            t0 = time.perf_counter()
            model = _CustomLayoutModel(sc, lay)
            result = model.run_simulation()
            wall = time.perf_counter() - t0
            agents = list(model.agents)
            exited = [a for a in agents if a.exited]
            t_evac = result.metrics.total_evacuation_time
            all_rows.append({
                "e3_width": width, "seed": seed,
                "t_evac": round(t_evac, 2),
                "n_evacuated": len(exited), "n_total": len(agents),
                "mean_flow": round(len(exited) / max(t_evac, 1.0), 4),
                "wall_time_s": round(wall, 2),
            })
            print(f"  seed={seed}  T_evac={t_evac:.1f}s  evac={len(exited)}/{len(agents)}")

    save_csv(all_rows, Path(__file__).parent.parent / "results" / "exp01_exit_width.csv")

