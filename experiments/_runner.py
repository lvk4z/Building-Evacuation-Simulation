"""Shared Monte Carlo runner used by all experiment scripts."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

from src.simulation.mesa_model import EvacuationModel
from src.simulation.scenario import BaselineScenario


def run_mc(
    scenario_factory: Callable[[int], BaselineScenario],
    seeds: list[int],
    extra_cols: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run multiple seeds for a given scenario factory and return rows."""
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        sc = scenario_factory(seed)
        sc.simulation.seed = seed
        model = EvacuationModel(sc)
        t0 = time.perf_counter()
        result = model.run_simulation()
        wall = time.perf_counter() - t0

        agents = list(model.agents)
        exited = [a for a in agents if a.exited]
        t_evac = result.metrics.total_evacuation_time
        mean_flow = len(exited) / max(t_evac, 1.0)  # agents/second

        row: dict[str, Any] = {
            "seed": seed,
            "t_evac": round(t_evac, 2),
            "n_evacuated": len(exited),
            "n_total": len(agents),
            "mean_flow": round(mean_flow, 4),
            "wall_time_s": round(wall, 2),
        }
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
        print(f"  seed={seed:3d}  T_evac={t_evac:6.1f}s  "
              f"evac={len(exited)}/{len(agents)}  flow={mean_flow:.3f} ag/s")
    return rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")
