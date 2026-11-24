"""Plot the archived mixed-scene simulation as a top-down view."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "historical" / "mixed-480hz" / "episode-412.json"
DEFAULT_OUTPUT = ROOT / "assets" / "figures" / "archived-mixed-trajectory.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as source:
        episode = json.load(source)

    ink = "#172638"
    secondary = "#455970"
    path_color = "#1456a0"
    motion_color = "#ad5322"
    footprint_fill = "#e0e7ed"
    footprint_outline = "#778a9d"
    goal_color = "#137859"

    fig, ax = plt.subplots(figsize=(6.8, 7.4), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.16, top=0.86)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(-8, 508), ylim=(-8, 508), xticks=range(0, 501, 100),
           yticks=range(0, 501, 100), xlabel="X position (m)", ylabel="Y position (m)")
    ax.set_axisbelow(True)
    ax.grid(color="#e6ecf2", linewidth=0.7)
    ax.tick_params(colors=secondary, labelsize=10)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    for spine in ax.spines.values():
        spine.set_color("#a0adba")

    moving_indices = {obstacle["index"] for obstacle in episode["dynamic_obstacles"]}
    for index, obstacle in enumerate(episode["Threaten"]):
        if index not in moving_indices:
            position = obstacle["position"]
            footprint = Circle((float(position["x"]), float(position["y"])),
                               float(obstacle["_R"]), facecolor=footprint_fill,
                               edgecolor=footprint_outline, linewidth=0.85, zorder=2)
            ax.add_patch(footprint)

    for obstacle in episode["dynamic_obstacles"]:
        _, x, y, _ = zip(*obstacle["trajectory_txyz"])
        ax.plot(x, y, color=motion_color, linewidth=1.15, linestyle="--", alpha=0.8, zorder=3)

    x, y, _ = zip(*episode["path"])
    (trajectory,) = ax.plot(x, y, color=path_color, linewidth=1.9, label="UAV trajectory", zorder=4)
    (start,) = ax.plot(x[0], y[0], "o", color=path_color, markeredgecolor="white",
                       markersize=8.5, label="Start", zorder=5)
    (finish,) = ax.plot(x[-1], y[-1], "o", markerfacecolor="white", markeredgecolor=path_color,
                        markeredgewidth=1.4, markersize=6.5, label="Recorded finish", zorder=5)
    goal_x, goal_y, _ = episode["goal_position_m"]
    (goal,) = ax.plot(goal_x, goal_y, "D", color=goal_color, markeredgecolor="white",
                      markersize=8, label="Goal", zorder=5)

    footprints = Patch(facecolor=footprint_fill, edgecolor=footprint_outline,
                       label="Stationary footprints")
    tracks = Line2D([], [], color=motion_color, linestyle="--", linewidth=1.15,
                    label="Moving obstacle tracks")
    fig.text(0.13, 0.953, "Archived mixed-scene trajectory", fontsize=19,
             color=ink, fontweight="bold", va="top")
    fig.text(0.13, 0.905,
             f"Episode {episode['epoch']}  |  {episode['timing']['control_hz']} Hz  |  XY projection",
             fontsize=11, color=secondary, va="top")
    fig.legend(handles=[trajectory, footprints, start, tracks, goal, finish],
               loc="lower center", bbox_to_anchor=(0.52, 0.027), ncol=3,
               columnspacing=1.4, frameon=False, fontsize=10, labelcolor=ink)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
