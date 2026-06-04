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
from src.simulation.scenario import (
    load_baseline_scenario,
)

# ── Colours ───────────────────────────────────────────────────────────────────
_STATE_COLORS = {
    "waiting":   "#d4a72c",
    "moving":    "#2b6cb0",
    "congested": "#c53030",
    "evacuated": "#2f855a",
    "leader":    "#805ad5",
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

    # 2. Smoke overlay (Scenario B) — density grows with current_time
    if model.smoke is not None:
        # Use the actual density (updated every step) so visual matches navigation
        vis_density = model.smoke.density
        smoke_rgba = np.zeros((*vis_density.shape, 4), dtype=float)
        smoke_rgba[..., 0] = 0.95  # red
        smoke_rgba[..., 1] = 0.25  # green
        smoke_rgba[..., 2] = 0.0   # blue
        smoke_rgba[..., 3] = np.clip(vis_density * 1.4, 0.0, 0.88)
        ax.imshow(
            smoke_rgba,
            origin="lower",
            extent=(0.0, model.layout.width_m, 0.0, model.layout.height_m),
            interpolation="bilinear",
            zorder=1,
        )
        # Fire origin marker — always shown even before smoke spreads
        ox, oy = model.smoke.origin_world
        ax.scatter([ox], [oy], s=180, c="black", marker="o",
                   edgecolors="#ff6600", linewidths=1.5, zorder=6, label="Fire")

    # 3. Exit segments + labels
    for exit_name, seg_start, seg_end in model.layout.iter_named_exit_segments():
        is_disabled = exit_name in model._disabled_exits
        color = "#888888" if is_disabled else _EXIT_COLORS.get(exit_name, "#2f855a")
        lw = 3 if is_disabled else 6
        ax.plot(
            [seg_start[0], seg_end[0]],
            [seg_start[1], seg_end[1]],
            color=color, linewidth=lw, solid_capstyle="round",
            linestyle="--" if is_disabled else "-",
            zorder=2,
        )
        exit_def = model.layout.exits[exit_name]
        offset = {
            "west":  np.array([ 1.6,  0.0]),
            "east":  np.array([-2.1,  0.0]),
            "south": np.array([ 0.0,  1.3]),
            "north": np.array([ 0.0, -1.4]),
        }[exit_def.side]
        lp = exit_def.center + offset
        label_text = f"{exit_name} ✗" if is_disabled else exit_name
        ax.text(
            lp[0], lp[1], label_text, fontsize=9,
            color="#666666" if is_disabled else "#1a202c",
            ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                  "edgecolor": color, "alpha": 0.95},
            zorder=3,
        )

    # 4. Agents (drawn on top of building)
    agents = tuple(model.agents)
    if agents:
        xs     = np.array([a.position[0] for a in agents])
        ys     = np.array([a.position[1] for a in agents])
        sizes  = np.array([max(8.0, a.radius * 260) for a in agents])
        colors = []
        for a in agents:
            if a.exited:
                colors.append(_STATE_COLORS["evacuated"])
            elif a.agent_type == "leader" and not a.exited:
                colors.append(_STATE_COLORS["leader"])
            elif not a.is_active(model.current_time):
                colors.append(_STATE_COLORS["waiting"])
            else:
                ratio = float(np.linalg.norm(a.velocity)) / max(a.desired_speed, 1e-6)
                colors.append(
                    _STATE_COLORS["congested"] if ratio < 0.35 else _STATE_COLORS["moving"]
                )
        # Split into regular agents and leaders for distinct rendering
        leader_mask = np.array([a.agent_type == "leader" for a in agents], dtype=bool)
        regular_mask = ~leader_mask

        if regular_mask.any():
            ax.scatter(
                xs[regular_mask], ys[regular_mask],
                s=sizes[regular_mask],
                c=[c for c, ldr in zip(colors, leader_mask) if not ldr],
                edgecolors="#1a202c", linewidths=0.3,
                alpha=0.88, zorder=5,
            )
        if leader_mask.any():
            # Leaders: larger star marker with white outline so they stand out
            ax.scatter(
                xs[leader_mask], ys[leader_mask],
                s=sizes[leader_mask] * 3.5,
                c=[c for c, ldr in zip(colors, leader_mask) if ldr],
                marker="*",
                edgecolors="white", linewidths=1.2,
                alpha=1.0, zorder=7,
            )

    # 5. Status box
    total = len(agents)
    evacuated = sum(1 for a in agents if a.exited)
    moving = sum(1 for a in agents if a.is_active(model.current_time) and not a.exited)
    waiting = total - evacuated - moving
    n_leaders = sum(1 for a in agents if a.agent_type == "leader")
    has_smoke = model.smoke is not None
    smoke_r = round(model.smoke.spread_rate * model.current_time, 1) if has_smoke else 0
    tags = []
    if has_smoke:
        tags.append(f"SMOKE r={smoke_r}m")
    if n_leaders > 0:
        tags.append(f"LEADERS \u2605 {n_leaders}")
    tag_str = "  ".join(tags) if tags else "BASELINE"
    ax.text(
        0.01, 0.99,
        f"[{tag_str}]  t = {model.current_time:.1f} s\n"
        f"evacuated: {evacuated}/{total}\n"
        f"moving:    {moving}  | waiting: {waiting}",
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "#cbd5e0", "alpha": 0.95},
        zorder=8,
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


# ── Interactive model class ──────────────────────────────────────────────────

class InteractiveEvacuationModel(EvacuationModel):
    """Accepts flat kwargs from SolaraViz sliders.

    *smoke_spread_rate* = 0  → no smoke (baseline).
    *smoke_spread_rate* > 0  → Scenario B with the given rate.
    *leader_fraction*   = 0  → all regular agents.
    *leader_fraction*   > 0  → that fraction of agents become leaders (★).
    Both can be active at the same time.
    """

    def __init__(
        self,
        n_agents: int = 50,
        desired_speed: float = 1.0,
        reaction_time_scale: float = 12.0,
        seed: int = 42,
        smoke_spread_rate: float = 0.0,
        leader_fraction: float = 0.0,
    ) -> None:
        from src.simulation.scenario import (
            LeaderConfig, ScenarioType, SmokeConfig,
        )

        sc = load_baseline_scenario(_CONFIG_PATH)

        if smoke_spread_rate > 0.0:
            # Pick a random interior walkable cell as the fire origin,
            # reproducible via the current seed.
            _rng = np.random.default_rng(int(seed) + 7919)  # offset avoids collision with agent seed
            _lt = BuildingLayout.baseline()
            _wy, _wx = np.where(_lt.walkable)
            # Keep only interior cells (away from building edges and exits)
            _margin = 4
            _mask = (
                (_wy > _margin) & (_wy < _lt.grid_height - _margin) &
                (_wx > _margin) & (_wx < _lt.grid_width  - _margin)
            )
            _iy, _ix = _wy[_mask], _wx[_mask]
            _pick = int(_rng.integers(0, len(_iy)))
            fire_x = float((_ix[_pick] + 0.5) * _lt.cell_size)
            fire_y = float((_iy[_pick] + 0.5) * _lt.cell_size)

            smoke_cfg = SmokeConfig(
                fire_origin_x=fire_x,
                fire_origin_y=fire_y,
                spread_rate=float(smoke_spread_rate),
                visibility_threshold=0.5,
                disabled_exits=[],
            )
            object.__setattr__(sc, "scenario_type", ScenarioType.SMOKE)
            object.__setattr__(sc, "smoke", smoke_cfg)

        if leader_fraction > 0.0:
            leader_cfg = LeaderConfig(
                fraction=float(leader_fraction),
                speed_mean=1.0,
                speed_std=0.1,
                reaction_time_scale=5.0,
            )
            if smoke_spread_rate <= 0.0:
                object.__setattr__(sc, "scenario_type", ScenarioType.LEADERS)
            object.__setattr__(sc, "leaders", leader_cfg)

        sc.agents.total = int(n_agents)
        sc.agents.desired_speed_mean = float(desired_speed)
        sc.agents.reaction_time_scale = float(reaction_time_scale)
        sc.simulation.seed = int(seed)
        super().__init__(sc)


# ── Interactive parameter sliders ─────────────────────────────────────────────

model_params = {
    "n_agents": Slider(
        label="Agents",
        value=50, min=20, max=300, step=10,
    ),
    "desired_speed": Slider(
        label="Desired speed [m/s]",
        value=1.0, min=0.5, max=4.0, step=0.1,
    ),
    "reaction_time_scale": Slider(
        label="Reaction time median [s]",
        value=12.0, min=1.0, max=60.0, step=1.0,
    ),
    "smoke_spread_rate": Slider(
        label="Smoke spread rate [m/s]  (0 = off)",
        value=0.0, min=0.0, max=0.8, step=0.1,
    ),
    "leader_fraction": Slider(
        label="Leader fraction ★  (0 = off)",
        value=0.0, min=0.0, max=0.30, step=0.05,
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

