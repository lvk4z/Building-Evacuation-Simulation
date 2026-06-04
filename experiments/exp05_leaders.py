"""Experiment 5 — Effect of leader agent fraction on T_evac.

Vary leader fraction: 0% / 5% / 10% / 20% / 30%.
190 agents total, 10 MC seeds each.

Run:
    python experiments/exp05_leaders.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.scenario import load_leaders_scenario
from experiments._runner import run_mc, save_csv

CONFIG = Path(__file__).parent.parent / "config.yaml"
SEEDS = list(range(10))
LEADER_FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30]

if __name__ == "__main__":
    all_rows = []
    for frac in LEADER_FRACTIONS:
        print(f"\n--- leader_fraction = {frac:.0%} ---")

        def factory(seed: int, f=frac):
            sc = load_leaders_scenario(CONFIG, leader_fraction=f)
            sc.simulation.seed = seed
            return sc

        rows = run_mc(factory, SEEDS, extra_cols={"leader_fraction": frac})
        all_rows.extend(rows)

    save_csv(all_rows, Path(__file__).parent.parent / "results" / "exp05_leaders.csv")
