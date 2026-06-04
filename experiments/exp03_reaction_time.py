"""Experiment 3 — Effect of reaction-time distribution on evacuation time percentiles.

Vary reaction_time_scale (median pre-evacuation delay): 1 / 5 / 10 / 20 / 40 s.
Baseline scenario, 190 agents, 15 MC seeds.

Run:
    python experiments/exp03_reaction_time.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.scenario import load_baseline_scenario
from experiments._runner import run_mc, save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
SEEDS = list(range(15))
RT_SCALES = [1.0, 5.0, 10.0, 20.0, 40.0]

if __name__ == "__main__":
    all_rows = []
    for rt in RT_SCALES:
        print(f"\n--- reaction_time_scale = {rt} s ---")

        def factory(seed: int, rt_scale=rt):
            sc = load_baseline_scenario(CONFIG)
            sc.agents.reaction_time_scale = rt_scale
            sc.simulation.seed = seed
            return sc

        rows = run_mc(factory, SEEDS, extra_cols={"reaction_time_scale": rt})
        all_rows.extend(rows)

    save_csv(all_rows, Path(__file__).parent.parent / "results" / "exp03_reaction_time.csv")
