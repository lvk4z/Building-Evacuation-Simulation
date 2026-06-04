"""Experiment 4 — Effect of smoke spread rate on evacuation time and route split.

Fire in the bottom-center room (x=18, y=4).  E2 (east) is pre-disabled.
Vary smoke spread_rate: 0.1 / 0.2 / 0.4 / 0.6 / 0.8 m/s.
190 agents, 10 MC seeds.

Run:
    python experiments/exp04_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.scenario import load_smoke_scenario
from experiments._runner import run_mc, save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
SEEDS = list(range(10))
SPREAD_RATES = [0.1, 0.2, 0.4, 0.6, 0.8]

if __name__ == "__main__":
    all_rows = []
    for rate in SPREAD_RATES:
        print(f"\n--- smoke spread_rate = {rate} m/s ---")

        def factory(seed: int, sr=rate):
            sc = load_smoke_scenario(CONFIG)
            sc.smoke.spread_rate = sr
            sc.simulation.seed = seed
            return sc

        rows = run_mc(factory, SEEDS, extra_cols={"spread_rate": rate, "scenario": "smoke"})
        all_rows.extend(rows)

    save_csv(all_rows, Path(__file__).parent.parent / "results" / "exp04_smoke.csv")
