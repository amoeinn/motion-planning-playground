"""Compare A*, Dijkstra, RRT, and RRT* on the same world, side by side.

Renders four plots showing how each algorithm explored the space:
  - A* and Dijkstra: graph search, differ only in heuristic
  - RRT: sampling-based tree, greedy parent selection
  - RRT*: sampling-based tree, optimal parent + rewiring

On this 4-connected grid all four find a 27-step path (grid geometry
bounds them). The exploration patterns and tree topologies differ.

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
from src.rrt import rrt, rrt_star


def render_search_result(ax, world, result, title: str, tree_edges=None) -> None:
    """Render one planner's result. tree_edges is only used for RRT/RRT*."""
    grid = np.zeros((world.rows, world.cols), dtype=int)
    for r, c in world.obstacles:
        grid[r, c] = 1
    for r, c in result.explored:
        if grid[r, c] == 0:
            grid[r, c] = 2

    cmap = plt.matplotlib.colors.ListedColormap(["white", "black", "#a8d8ea"])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=2)

    if tree_edges:
        for parent, child in tree_edges:
            ax.plot([parent[1], child[1]], [parent[0], child[0]],
                    color="gray", linewidth=0.6, alpha=0.7, zorder=2)

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

    ax.set_title(title)


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    astar_result = astar(world)
    dijkstra_result = dijkstra(world)
    rrt_result = rrt(world, seed=42)
    rrt_star_result = rrt_star(world, seed=42, max_iterations=1000)

    print(f"A*:       path length {astar_result.path_length}, "
          f"explored {len(astar_result.explored)} cells")
    print(f"Dijkstra: path length {dijkstra_result.path_length}, "
          f"explored {len(dijkstra_result.explored)} cells")
    print(f"RRT:      path length {rrt_result.path_length}, "
          f"nodes {len(rrt_result.explored)}, "
          f"iterations {rrt_result.iterations_used}")
    print(f"RRT*:     path length {rrt_star_result.path_length}, "
          f"nodes {len(rrt_star_result.explored)}, "
          f"iterations {rrt_star_result.iterations_used}")

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    render_search_result(axes[0], world, astar_result,
                         f"A* (Manhattan)\npath: {astar_result.path_length}   "
                         f"explored: {len(astar_result.explored)}")
    render_search_result(axes[1], world, dijkstra_result,
                         f"Dijkstra\npath: {dijkstra_result.path_length}   "
                         f"explored: {len(dijkstra_result.explored)}")
    render_search_result(axes[2], world, rrt_result,
                         f"RRT (seed=42)\npath: {rrt_result.path_length}   "
                         f"nodes: {len(rrt_result.explored)}   "
                         f"iter: {rrt_result.iterations_used}",
                         tree_edges=rrt_result.tree_edges)
    render_search_result(axes[3], world, rrt_star_result,
                         f"RRT* (seed=42, 1000 iter)\n"
                         f"path: {rrt_star_result.path_length}   "
                         f"nodes: {len(rrt_star_result.explored)}",
                         tree_edges=rrt_star_result.tree_edges)

    plt.suptitle("A* vs Dijkstra vs RRT vs RRT* on the same world",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()