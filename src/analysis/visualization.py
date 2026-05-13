from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.simulation.mesa_model import FrameSnapshot, SimulationResult


STATE_COLORS = {
    "waiting": "#d4a72c",
    "moving": "#2b6cb0",
    "congested": "#c53030",
    "evacuated": "#2f855a",
}
EXIT_COLORS = {
    "E1": "#2b6cb0",
    "E2": "#805ad5",
    "E3": "#dd6b20",
    "E4": "#319795",
}


def _draw_layout(ax: plt.Axes, result: SimulationResult) -> None:
    """Draw the floor plan with stronger contrast and exit labels."""

    layout = result.layout
    ax.set_facecolor("#fcfcfc")
    ax.imshow(
        (~layout.walkable).astype(float),
        origin="lower",
        extent=(0.0, layout.width_m, 0.0, layout.height_m),
        cmap="binary",
        alpha=0.92,
        interpolation="nearest",
    )

    for exit_name, start, end in layout.iter_named_exit_segments():
        exit_def = layout.exits[exit_name]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#2f855a",
            linewidth=5.0,
            solid_capstyle="round",
            zorder=3,
        )

        label_offset = {
            "west": np.array([1.4, 0.0]),
            "east": np.array([-2.0, 0.0]),
            "south": np.array([0.0, 1.1]),
            "north": np.array([0.0, -1.2]),
        }[exit_def.side]
        label_pos = exit_def.center + label_offset
        ax.text(
            label_pos[0],
            label_pos[1],
            exit_name,
            fontsize=10,
            color="#1a202c",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#2f855a", "alpha": 0.95},
            zorder=4,
        )

    ax.set_xlim(0.0, layout.width_m)
    ax.set_ylim(0.0, layout.height_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Baseline evacuation simulation", fontsize=15, weight="bold")


def _has_interactive_backend() -> bool:
    """Return whether matplotlib can open an interactive window."""

    backend = matplotlib.get_backend().lower()
    non_interactive_markers = ("agg", "pdf", "ps", "svg", "template")
    return not any(marker in backend for marker in non_interactive_markers)


def _agent_sizes(result: SimulationResult) -> np.ndarray:
    """Convert physical body radius to plot marker area."""

    radii = np.array([agent.radius for agent in result.agents], dtype=float)
    return np.square(radii * 58.0)


def _agent_colors(frame: FrameSnapshot, result: SimulationResult) -> np.ndarray:
    """Color agents by state: waiting=yellow, moving=blue, congested=red, evacuated=green."""

    desired_speeds = np.array([agent.desired_speed for agent in result.agents], dtype=float)
    colors: list[str] = []

    for index in range(len(result.agents)):
        if frame.exited_mask[index]:
            colors.append(STATE_COLORS["evacuated"])
            continue
        if not frame.active_mask[index]:
            colors.append(STATE_COLORS["waiting"])
            continue

        speed_ratio = frame.speeds[index] / max(desired_speeds[index], 1e-6)
        if speed_ratio < 0.35:
            colors.append(STATE_COLORS["congested"])
        else:
            colors.append(STATE_COLORS["moving"])

    return np.array(colors, dtype=object)


def _format_status_text(frame: FrameSnapshot) -> str:
    """Compose the main status panel text."""

    evacuated = int(frame.exited_mask.sum())
    moving = int((frame.active_mask & ~frame.exited_mask).sum())
    waiting = int((~frame.active_mask & ~frame.exited_mask).sum())
    active_speeds = frame.speeds[frame.active_mask & ~frame.exited_mask]
    mean_speed = float(np.mean(active_speeds)) if active_speeds.size else 0.0

    return (
        f"time: {frame.time:6.2f} s\n"
        f"evacuated: {evacuated:3d}\n"
        f"moving: {moving:3d}\n"
        f"waiting: {waiting:3d}\n"
        f"mean speed: {mean_speed:4.2f} m/s"
    )


def _format_exit_usage(frame: FrameSnapshot) -> str:
    """Compose a compact per-exit assignment summary."""

    parts = ["assigned exits:"]
    active_mask = ~frame.exited_mask
    for exit_name in sorted(EXIT_COLORS):
        count = int(np.sum((frame.exit_names == exit_name) & active_mask))
        parts.append(f"{exit_name}: {count:3d}")
    return "\n".join(parts)


def _build_legends(ax: plt.Axes) -> None:
    """Add state and exit legends to the plot."""

    state_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=color, markeredgecolor="#1a202c", label=label)
        for label, color in (
            ("waiting", STATE_COLORS["waiting"]),
            ("moving", STATE_COLORS["moving"]),
            ("congested", STATE_COLORS["congested"]),
            ("evacuated", STATE_COLORS["evacuated"]),
        )
    ]
    exit_handles = [
        Line2D([0], [0], color=color, linewidth=4, label=exit_name)
        for exit_name, color in EXIT_COLORS.items()
    ]

    state_legend = ax.legend(handles=state_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="agent state")
    ax.add_artist(state_legend)
    ax.legend(handles=exit_handles, loc="upper left", bbox_to_anchor=(1.01, 0.57), frameon=True, title="exit labels")


def _draw_frame(ax: plt.Axes, result: SimulationResult, frame_index: int) -> None:
    """Draw a single clean frame for snapshot export."""

    frame = result.frames[frame_index]
    sizes = _agent_sizes(result)
    colors = _agent_colors(frame, result)

    _draw_layout(ax, result)
    ax.scatter(
        frame.positions[:, 0],
        frame.positions[:, 1],
        s=sizes,
        c=colors.tolist(),
        edgecolors="#1a202c",
        linewidths=0.25,
        alpha=0.9,
        zorder=5,
    )

    ax.text(
        1.02,
        0.38,
        _format_status_text(frame),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.98},
    )
    ax.text(
        1.02,
        0.15,
        _format_exit_usage(frame),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.98},
    )
    _build_legends(ax)


def save_simulation_snapshot(result: SimulationResult, output_path: str | Path, frame_index: int = -1) -> Path:
    """Save a single frame when interactive animation is unavailable."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    _draw_frame(ax, result, frame_index)
    plt.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def animate_simulation(
    result: SimulationResult,
    interval_ms: int = 50,
    trail_length: int = 8,
    save_path: str | None = None,
) -> animation.FuncAnimation | None:
    """Create a clean research-style animation for the recorded frames.

    The `trail_length` parameter is currently unused and kept for CLI
    compatibility with the earlier alpha version.
    """

    del trail_length
    interactive_backend = _has_interactive_backend()

    if not interactive_backend and save_path is None:
        preview_path = save_simulation_snapshot(result, Path("outputs") / "baseline_preview.png")
        print(f"Non-interactive matplotlib backend detected. Saved preview to {preview_path}.")
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    _draw_layout(ax, result)
    _build_legends(ax)

    scatter = ax.scatter([], [], s=[], c=[], edgecolors="#1a202c", linewidths=0.25, alpha=0.9, zorder=5)
    status_text = ax.text(
        1.02,
        0.38,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.98},
    )
    exit_text = ax.text(
        1.02,
        0.15,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.98},
    )
    summary_text = ax.text(
        0.02,
        1.02,
        f"completed: {'yes' if result.metrics.completed else 'no'} | total evacuation time: {result.metrics.total_evacuation_time:6.2f} s",
        transform=ax.transAxes,
        fontsize=11,
        color="#1a202c",
    )

    sizes = _agent_sizes(result)

    def update(frame_index: int):
        frame = result.frames[frame_index]
        scatter.set_offsets(frame.positions)
        scatter.set_sizes(sizes)
        scatter.set_color(_agent_colors(frame, result).tolist())
        status_text.set_text(_format_status_text(frame))
        exit_text.set_text(_format_exit_usage(frame))
        return [scatter, status_text, exit_text, summary_text]

    anim = animation.FuncAnimation(fig, update, frames=len(result.frames), interval=interval_ms, blit=False, repeat=False)

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path)

    plt.tight_layout()
    if interactive_backend:
        plt.show()
    else:
        plt.close(fig)
    return anim


def plot_evacuation_curve(result: SimulationResult) -> None:
    """Plot cumulative number of evacuated agents over time."""

    exit_times = sorted(agent.exit_time for agent in result.agents if agent.exit_time is not None)
    if not exit_times:
        return

    cumulative = np.arange(1, len(exit_times) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(exit_times, cumulative, where="post", color="#2b6cb0", linewidth=2.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("evacuated agents")
    ax.set_title("Cumulative evacuation curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if _has_interactive_backend():
        plt.show()
    else:
        output_path = Path("outputs") / "evacuation_curve.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Non-interactive matplotlib backend detected. Saved evacuation curve to {output_path}.")
