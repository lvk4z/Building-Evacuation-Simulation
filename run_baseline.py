from __future__ import annotations

import argparse
from pathlib import Path

from src.simulation.mesa_model import EvacuationModel
from src.simulation.scenario import load_baseline_scenario


def parse_args() -> argparse.Namespace:
    """Parse command-line switches for the headless baseline runner."""

    parser = argparse.ArgumentParser(description="Run the baseline evacuation simulation in headless mode.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline scenario and print summary metrics."""

    args = parse_args()
    scenario = load_baseline_scenario(args.config)
    model = EvacuationModel(scenario)
    result = model.run_simulation()

    print("Baseline simulation finished")
    print(f"  completed          : {result.metrics.completed}")
    print(f"  evacuated agents   : {result.metrics.evacuated_agents}/{result.metrics.total_agents}")
    print(f"  total evac time    : {result.metrics.total_evacuation_time:.2f} s")
    print(f"  mean exit time     : {result.metrics.mean_exit_time:.2f} s")
    print(f"  median exit time   : {result.metrics.median_exit_time:.2f} s")


if __name__ == "__main__":
    main()
