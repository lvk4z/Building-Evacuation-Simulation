"""Experiment 2 — Effect of crowd density (agent count) on T_evac and congestion.

Vary total agent count: 50 / 100 / 150 / 190 / 250 / 300.
Baseline scenario, 10 MC seeds each.

Run:
    python experiments/exp02_density.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.scenario import load_baseline_scenario
from experiments._runner import run_mc, save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
SEEDS = list(range(10))
DENSITIES = [50, 100, 150, 190, 250, 300]

if __name__ == "__main__":
    all_rows = []
    for n in DENSITIES:
        print(f"\n--- n_agents = {n} ---")

        def factory(seed: int, n_agents=n):
            sc = load_baseline_scenario(CONFIG)
            sc.agents.total = n_agents
            sc.simulation.seed = seed
            return sc

        rows = run_mc(factory, SEEDS, extra_cols={"n_agents": n})
        all_rows.extend(rows)

    save_csv(all_rows, Path(__file__).parent.parent / "results" / "exp02_density.csv")
