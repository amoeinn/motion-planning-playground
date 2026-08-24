"""Compare A* against Dijkstra on the same world, side by side.

Renders two plots showing how each algorithm explored the space. Both find
the optimal path, but the exploration patterns are dramatically different.

Usage:
    python examples/run_comparison_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.astar import astar, dijkstra
from src.grid_world import make_wall_world


def render_result(ax, world, result, title: str) -> None:
    """Render a single algorithm's result onto the given axis."""
    grid = np.zeros((world.rows, world.cols), dtype=int)
    for r, c in world.obstacles:
        grid[r, c] = 1
    for r, c in result.explored:
        if grid[r, c] == 0:
            grid[r, c] = 2

    cmap = plt.matplotlib.colors.ListedColormap(["white", "black", "#a8d8ea"])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=2)

    if result.path:
        rows = [c[0] for c in result.path]
        cols = [c[1] for c in result.path]
        ax.plot(cols, rows, color="red", linewidth=2.5, zorder=3)

    ax.plot(world.start[1], world.start[0], marker="s", color="green",
            markersize=12, zorder=4)
    ax.plot(world.goal[1], world.goal[0], marker="*", color="gold",
            markersize=18, markeredgecolor="black", zorder=4)

    ax.set_xticks(np.arange(-0.5, world.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    stats = f"path: {result.path_length}   explored: {len(result.explored)}"
    ax.set_title(f"{title}\n{stats}")


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    astar_result = astar(world)
    dijkstra_result = dijkstra(world)

    print(f"A*:       path length {astar_result.path_length}, "
          f"explored {len(astar_result.explored)} cells")
    print(f"Dijkstra: path length {dijkstra_result.path_length}, "
          f"explored {len(dijkstra_result.explored)} cells")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    render_result(ax1, world, astar_result, "A* (Manhattan heuristic)")
    render_result(ax2, world, dijkstra_result, "Dijkstra (no heuristic)")

    plt.suptitle("A* vs Dijkstra on the same world", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
