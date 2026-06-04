"""Analyze all experiment results and produce a summary report with statistics.

Reads CSVs from results/ and prints a markdown-formatted summary with:
- Mean ± std of T_evac per parameter level
- Approximate 95% CI via bootstrap (fast: 1000 samples)
- Simple ASCII trend description

Run:
    python experiments/analyze_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
REPORT_PATH = RESULTS_DIR / "summary_report.md"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    boot_means = np.array([
        np.mean(arr[np.random.randint(0, len(arr), len(arr))])
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def summarize(rows: list[dict], group_col: str, value_col: str = "t_evac") -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row[group_col]].append(float(row[value_col]))

    summary = []
    for key in sorted(groups, key=lambda k: float(k)):
        vals = groups[key]
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        lo, hi = bootstrap_ci(vals)
        summary.append({
            "param": float(key),
            "n": len(vals),
            "mean": round(mu, 1),
            "std": round(sd, 1),
            "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
            "p50": round(float(np.percentile(vals, 50)), 1),
            "p90": round(float(np.percentile(vals, 90)), 1),
        })
    return summary


def table_md(rows: list[dict], header: str) -> str:
    if not rows:
        return "_No data_\n"
    cols = list(rows[0].keys())
    header_line = "| " + " | ".join(str(c) for c in cols) + " |"
    sep_line = "| " + " | ".join("---" for _ in cols) + " |"
    data_lines = [
        "| " + " | ".join(str(r[c]) for c in cols) + " |"
        for r in rows
    ]
    return header + "\n" + header_line + "\n" + sep_line + "\n" + "\n".join(data_lines) + "\n"


sections: list[str] = ["# Evacuation Simulation — Experiment Results Summary\n"]

EXPERIMENTS = [
    ("exp01_exit_width.csv",   "e3_width",         "## E1 — Exit width (E3) effect on T_evac [s]",
     "Wider exits reduce T_evac through increased flow capacity at the north bottleneck."),
    ("exp02_density.csv",      "n_agents",          "## E2 — Crowd density (agent count) effect on T_evac [s]",
     "Higher density increases T_evac nonlinearly due to social-force congestion."),
    ("exp03_reaction_time.csv","reaction_time_scale","## E3 — Reaction-time scale effect on T_evac [s]",
     "Longer reaction times shift T_evac upward; P90 is more sensitive than P50."),
    ("exp04_smoke.csv",        "spread_rate",       "## E4 — Smoke spread rate effect on T_evac [s]",
     "Faster smoke spread blocks more corridors, forcing detours and raising T_evac."),
    ("exp05_leaders.csv",      "leader_fraction",   "## E5 — Leader fraction effect on T_evac [s]",
     "Leaders with short reaction times and optimal exit knowledge reduce T_evac."),
]

for fname, group_col, heading, interpretation in EXPERIMENTS:
    csv_path = RESULTS_DIR / fname
    if not csv_path.exists():
        sections.append(f"{heading}\n\n_Not yet run (missing {fname})_\n")
        continue

    rows = read_csv(csv_path)
    summary = summarize(rows, group_col)
    sections.append(table_md(summary, heading))
    sections.append(f"**Interpretation:** {interpretation}\n")

    # Print to console as well
    print(f"\n{heading}")
    print(f"{'param':>8}  {'mean':>7}  {'std':>6}  {'CI 95%':>14}  {'P50':>7}  {'P90':>7}")
    for r in summary:
        print(f"{r['param']:>8}  {r['mean']:>7}  {r['std']:>6}  "
              f"[{r['ci_lo']}, {r['ci_hi']}]  {r['p50']:>7}  {r['p90']:>7}")

report = "\n".join(sections)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report, encoding="utf-8")
print(f"\nReport saved to {REPORT_PATH}")
