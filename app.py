"""Interactive Solara visualization for the Building Evacuation Simulation.

Run with:
    solara run app.py

Then open http://localhost:8765 in your browser.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import solara
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

from src.environment.building import BuildingLayout
from src.simulation.mesa_model import EvacuationModel
from src.simulation.scenario import load_baseline_scenario

# ── Colours ───────────────────────────────────────────────────────────────────
_STATE_COLORS = {
    "waiting":   "#d4a72c",
    "moving":    "#2b6cb0",
    "congested": "#c53030",
    "evacuated": "#2f855a",
}
_EXIT_COLORS = {
    "E1": "#2b6cb0",
    "E2": "#805ad5",
    "E3": "#dd6b20",
    "E4": "#319795",
}

_layout = BuildingLayout.baseline()
_CONFIG_PATH = Path(__file__).with_name("config.yaml")

_wall_img = (~_layout.walkable).astype(float)


# ── Custom space component ─────────────────────────────────────────────────────

@solara.component
def EvacuationSpaceComponent(model: EvacuationModel) -> None:
    update_counter.get()

    fig = Figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    # 1. Building walls (drawn first = behind everything)
    ax.imshow(
        _wall_img,
        origin="lower",
        extent=(0.0, _layout.width_m, 0.0, _layout.height_m),
        cmap="Greys",
        vmin=0, vmax=1,
        alpha=0.6,
        interpolation="nearest",
        zorder=0,
    )

    # 2. Exit segments + labels
    for exit_name, seg_start, seg_end in model.layout.iter_named_exit_segments():
        color = _EXIT_COLORS.get(exit_name, "#2f855a")
        ax.plot(
            [seg_start[0], seg_end[0]],
            [seg_start[1], seg_end[1]],
            color=color, linewidth=6, solid_capstyle="round", zorder=1,
        )
        exit_def = model.layout.exits[exit_name]
        offset = {
            "west":  np.array([ 1.6,  0.0]),
            "east":  np.array([-2.1,  0.0]),
            "south": np.array([ 0.0,  1.3]),
            "north": np.array([ 0.0, -1.4]),
        }[exit_def.side]
        lp = exit_def.center + offset
        ax.text(
            lp[0], lp[1], exit_name, fontsize=9,
            color="#1a202c", ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": color, "alpha": 0.95},
            zorder=2,
        )

    # 3. Agents (drawn on top of building)
    agents = tuple(model.agents)
    if agents:
        xs     = np.array([a.position[0] for a in agents])
        ys     = np.array([a.position[1] for a in agents])
        sizes  = np.array([max(8.0, a.radius * 260) for a in agents])
        colors = []
        for a in agents:
            if a.exited:
                colors.append(_STATE_COLORS["evacuated"])
            elif not a.is_active(model.current_time):
                colors.append(_STATE_COLORS["waiting"])
            else:
                ratio = float(np.linalg.norm(a.velocity)) / max(a.desired_speed, 1e-6)
                colors.append(
                    _STATE_COLORS["congested"] if ratio < 0.35 else _STATE_COLORS["moving"]
                )
        ax.scatter(xs, ys, s=sizes, c=colors,
                   edgecolors="#1a202c", linewidths=0.3,
                   alpha=0.88, zorder=3)

    # 4. Status box
    total = len(agents)
    evacuated = sum(1 for a in agents if a.exited)
    moving = sum(1 for a in agents if a.is_active(model.current_time) and not a.exited)
    waiting = total - evacuated - moving
    ax.text(
        0.01, 0.99,
        f"t = {model.current_time:.1f} s\n"
        f"evacuated: {evacuated}/{total}\n"
        f"moving:    {moving}\n"
        f"waiting:   {waiting}",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "#cbd5e0", "alpha": 0.95},
        zorder=5,
    )

    ax.set_xlim(0, model.layout.width_m)
    ax.set_ylim(0, model.layout.height_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=9)
    ax.set_ylabel("y [m]", fontsize=9)
    ax.set_title("Building Evacuation — Social Force Model (Mesa)",
                 fontsize=11, weight="bold")

    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")

# ── Plot post-processors ───────────────────────────────────────────────────────

def post_process_evacuated(ax: Axes) -> None:
    ax.set_xlabel("step")
    ax.set_ylabel("agents")
    ax.set_title("Evacuated (green) vs Remaining (red)", fontsize=10, weight="bold")
    ax.grid(alpha=0.3)


def post_process_speed(ax: Axes) -> None:
    ax.set_xlabel("step")
    ax.set_ylabel("mean speed [m/s]")
    ax.set_title("Mean active-agent speed", fontsize=10, weight="bold")
    ax.grid(alpha=0.3)


# ── Interactive model class ────────────────────────────────────────────────────

class InteractiveEvacuationModel(EvacuationModel):
    """Thin subclass that accepts flat kwargs from SolaraViz sliders."""

    def __init__(
        self,
        n_agents: int = 50,
        desired_speed: float = 1.0,
        reaction_time_scale: float = 12.0,
        seed: int = 42,
    ) -> None:
        scenario = load_baseline_scenario(_CONFIG_PATH)
        scenario.agents.total = int(n_agents)
        scenario.agents.desired_speed_mean = float(desired_speed)
        scenario.agents.reaction_time_scale = float(reaction_time_scale)
        scenario.simulation.seed = int(seed)
        super().__init__(scenario)


# ── Interactive parameter sliders ─────────────────────────────────────────────

model_params = {
    "n_agents": Slider(
        label="Number of agents",
        value=50, min=20, max=350, step=10,
    ),
    "desired_speed": Slider(
        label="Desired speed [m/s]",
        value=1.0, min=0.5, max=4.0, step=0.1,
    ),
    "reaction_time_scale": Slider(
        label="Reaction time median [s]",
        value=12.0, min=1.0, max=60.0, step=1.0,
    ),
    "seed": Slider(
        label="Random seed",
        value=42, min=0, max=999, step=1,
    ),
}


# ── SolaraViz app ──────────────────────────────────────────────────────────────

page = SolaraViz(
    model=InteractiveEvacuationModel(),
    components=[
        EvacuationSpaceComponent,
        make_plot_component(
            measure={"evacuated": "#2f855a", "remaining": "#c53030"},
            post_process=post_process_evacuated,
        ),
    ],
    model_params=model_params,
    name="Building Evacuation Simulation",
    play_interval=100,
)
