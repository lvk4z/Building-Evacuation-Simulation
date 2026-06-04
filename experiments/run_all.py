"""Master experiment runner — executes all 5 experiments sequentially.

Tuned for fast turnaround: 5 seeds per configuration, max_time=120 s.
Saves results to results/*.csv then runs analyze_results.py.

Run:
    python experiments/run_all.py
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
from src.simulation.scenario import (
    load_baseline_scenario,
    load_smoke_scenario,
    load_leaders_scenario,
)
from experiments._runner import save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
RESULTS = Path(__file__).parent.parent / "results"
SEEDS = list(range(5))    # 5 seeds per configuration
MAX_TIME = 60.0           # cap at 60 s — enough for T_evac with 50–150 agents
N_AGENTS_SWEEP = 50       # agents used in E1/E3/E4/E5 parameter sweeps


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_scenario(sc, seed: int) -> dict:
    sc.simulation.seed = seed
    sc.simulation.max_time = MAX_TIME
    t0 = time.perf_counter()
    model = EvacuationModel(sc)
    result = model.run_simulation()
    wall = time.perf_counter() - t0
    agents = list(model.agents)
    exited = [a for a in agents if a.exited]
    t_evac = result.metrics.total_evacuation_time
    return {
        "seed": seed,
        "t_evac": round(t_evac, 2),
        "n_evacuated": len(exited),
        "n_total": len(agents),
        "mean_flow": round(len(exited) / max(t_evac, 1.0), 4),
        "wall_time_s": round(wall, 2),
    }


def _build_layout_e3(e3_width: float) -> BuildingLayout:
    lay = BuildingLayout(36.0, 20.0, 0.4)
    lay.walkable[0, :] = lay.walkable[-1, :] = False
    lay.walkable[:, 0] = lay.walkable[:, -1] = False
    lay._add_horizontal_wall(8.0,  [(5.1, 6.9), (17.1, 18.9), (29.1, 30.9)])
    lay._add_horizontal_wall(12.0, [(3.6, 5.4), (12.6, 14.4), (21.6, 23.4), (30.6, 32.4)])
    for x in (9.0, 18.0, 27.0):
        lay._add_vertical_wall(x, 12.0, 20.0, [])
    for x in (12.0, 24.0):
        lay._add_vertical_wall(x, 0.0, 8.0, [])
    lay._fill_rectangle(13.6, 15.0, 8.9, 11.1, walkable=False)
    lay._fill_rectangle(21.2, 22.6, 8.9, 11.1, walkable=False)
    lay._register_exit("E1", "west",  10.0, 2.0)
    lay._register_exit("E2", "east",  10.0, 1.2)
    lay._register_exit("E3", "north", 22.5, e3_width)
    lay._register_exit("E4", "south", 18.0, 1.8)
    lay.spawn_zones = [
        SpawnZone("top_left_1",    1.0,  8.0, 13.0, 19.0),
        SpawnZone("top_left_2",   10.0, 17.0, 13.0, 19.0),
        SpawnZone("top_right_1",  19.0, 26.0, 13.0, 19.0),
        SpawnZone("top_right_2",  28.0, 35.0, 13.0, 19.0),
        SpawnZone("bottom_left",   1.0, 11.0,  1.0,  7.0),
        SpawnZone("bottom_center",13.0, 23.0,  1.0,  7.0),
        SpawnZone("bottom_right", 25.0, 35.0,  1.0,  7.0),
    ]
    return lay


class _CustomLayoutModel(EvacuationModel):
    """EvacuationModel with injected layout (for exit-width experiment)."""

    def __init__(self, scenario, layout: BuildingLayout) -> None:
        import mesa
        mesa.Model.__init__(self, rng=scenario.simulation.seed)

        self.layout = layout
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
            agent_reporters={"exited": "exited", "agent_type": "agent_type"},
        )
        self.frames = []
        self._record_frame()
        self.datacollector.collect(self)


def _run_custom(sc, layout: BuildingLayout, seed: int) -> dict:
    sc.simulation.seed = seed
    sc.simulation.max_time = MAX_TIME
    t0 = time.perf_counter()
    model = _CustomLayoutModel(sc, layout)
    result = model.run_simulation()
    wall = time.perf_counter() - t0
    agents = list(model.agents)
    exited = [a for a in agents if a.exited]
    t_evac = result.metrics.total_evacuation_time
    return {
        "seed": seed,
        "t_evac": round(t_evac, 2),
        "n_evacuated": len(exited),
        "n_total": len(agents),
        "mean_flow": round(len(exited) / max(t_evac, 1.0), 4),
        "wall_time_s": round(wall, 2),
    }


# ── E1: exit width ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 1 — Exit E3 width")
print("=" * 60)
E3_WIDTHS = [0.8, 1.0, 1.2, 1.5, 2.0]
rows_e1 = []
for width in E3_WIDTHS:
    print(f"\n  E3 width = {width} m")
    for seed in SEEDS:
        lay = _build_layout_e3(width)
        sc = load_baseline_scenario(CONFIG)
        sc.agents.total = N_AGENTS_SWEEP
        row = _run_custom(sc, lay, seed)
        row["e3_width"] = width
        rows_e1.append(row)
        print(f"    seed={seed}  T_evac={row['t_evac']:6.1f}s  evac={row['n_evacuated']}/{row['n_total']}")
save_csv(rows_e1, RESULTS / "exp01_exit_width.csv")


# ── E2: density ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 2 — Crowd density (agent count)")
print("=" * 60)
AGENT_COUNTS = [30, 60, 100, 150, 200]
rows_e2 = []
for n in AGENT_COUNTS:
    print(f"\n  n_agents = {n}")
    for seed in SEEDS:
        sc = load_baseline_scenario(CONFIG)
        sc.agents.total = n
        row = _run_scenario(sc, seed)
        row["n_agents"] = n
        rows_e2.append(row)
        print(f"    seed={seed}  T_evac={row['t_evac']:6.1f}s  evac={row['n_evacuated']}/{row['n_total']}")
save_csv(rows_e2, RESULTS / "exp02_density.csv")


# ── E3: reaction time ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 3 — Reaction time scale")
print("=" * 60)
RT_SCALES = [1.0, 5.0, 10.0, 20.0, 40.0]
rows_e3 = []
for rt in RT_SCALES:
    print(f"\n  reaction_time_scale = {rt} s")
    for seed in SEEDS:
        sc = load_baseline_scenario(CONFIG)
        sc.agents.total = N_AGENTS_SWEEP
        sc.agents.reaction_time_scale = rt
        row = _run_scenario(sc, seed)
        row["reaction_time_scale"] = rt
        rows_e3.append(row)
        print(f"    seed={seed}  T_evac={row['t_evac']:6.1f}s  evac={row['n_evacuated']}/{row['n_total']}")
save_csv(rows_e3, RESULTS / "exp03_reaction_time.csv")


# ── E4: smoke ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 4 — Smoke spread rate")
print("=" * 60)
SPREAD_RATES = [0.1, 0.2, 0.4, 0.6, 0.8]
rows_e4 = []
for rate in SPREAD_RATES:
    print(f"\n  smoke spread_rate = {rate} m/s")
    for seed in SEEDS:
        sc = load_smoke_scenario(CONFIG)
        sc.agents.total = N_AGENTS_SWEEP
        sc.smoke.spread_rate = rate
        row = _run_scenario(sc, seed)
        row["spread_rate"] = rate
        rows_e4.append(row)
        print(f"    seed={seed}  T_evac={row['t_evac']:6.1f}s  evac={row['n_evacuated']}/{row['n_total']}")
save_csv(rows_e4, RESULTS / "exp04_smoke.csv")


# ── E5: leaders ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EXPERIMENT 5 — Leader agent fraction")
print("=" * 60)
LEADER_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30]
rows_e5 = []
for frac in LEADER_FRACS:
    print(f"\n  leader_fraction = {frac:.0%}")
    for seed in SEEDS:
        sc = load_leaders_scenario(CONFIG, leader_fraction=frac)
        sc.agents.total = N_AGENTS_SWEEP
        row = _run_scenario(sc, seed)
        row["leader_fraction"] = frac
        rows_e5.append(row)
        print(f"    seed={seed}  T_evac={row['t_evac']:6.1f}s  evac={row['n_evacuated']}/{row['n_total']}")
save_csv(rows_e5, RESULTS / "exp05_leaders.csv")


# ── analyze ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
import subprocess
subprocess.run([sys.executable, "experiments/analyze_results.py"], check=True)
