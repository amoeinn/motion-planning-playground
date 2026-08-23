"""Matplotlib visualization of grid worlds and planner results.

Renders a GridWorld with obstacles, start, goal, explored cells, and the
final path. Designed to make it easy to compare planner behavior visually.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .astar import AStarResult
from .grid_world import GridWorld


def plot_result(
    world: GridWorld,
    result: AStarResult,
    title: str = "A* Search Result",
    show: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """Render the world, explored cells, and final path.

    Colors:
      white  = free cell
      black  = obstacle
      light blue = cell explored by planner
      red line = final path
      green square = start
      gold star = goal
    """
    # Build a numeric grid: 0 = free, 1 = obstacle, 2 = explored
    grid = np.zeros((world.rows, world.cols), dtype=int)
    for r, c in world.obstacles:
        grid[r, c] = 1
    for r, c in result.explored:
        if grid[r, c] == 0:  # do not overwrite obstacles
            grid[r, c] = 2

    # Colormap: 0=white, 1=black, 2=light blue
    cmap = plt.matplotlib.colors.ListedColormap(["white", "black", "#a8d8ea"])

    fig, ax = plt.subplots(figsize=(world.cols * 0.5, world.rows * 0.5))
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=2)

    # Draw the path as a red line connecting cell centers
    if result.path:
        path_rows = [cell[0] for cell in result.path]
        path_cols = [cell[1] for cell in result.path]
        ax.plot(path_cols, path_rows, color="red", linewidth=2.5, zorder=3)

    # Start and goal markers
    ax.plot(world.start[1], world.start[0], marker="s", color="green",
            markersize=14, zorder=4, label="Start")
    ax.plot(world.goal[1], world.goal[0], marker="*", color="gold",
            markersize=20, markeredgecolor="black", zorder=4, label="Goal")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, world.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Title with stats
    stats = f"path length: {result.path_length}   explored: {len(result.explored)}"
    ax.set_title(f"{title}\n{stats}")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
