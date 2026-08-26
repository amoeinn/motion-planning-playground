"""Animated replay of planner searches on a grid world.

Both AStarResult and RRTResult already record their search in time order:
AStarResult.frontier_history is the sequence of cells popped from the open
set, and RRTResult.tree_edges is in insertion order. These functions replay
that record frame by frame so the search can be watched rather than only
inspected at its end state.
"""

from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .astar import AStarResult
from .grid_world import Cell, GridWorld
from .rrt import RRTResult

# 0 = free, 1 = obstacle, 2 = explored
CMAP = plt.matplotlib.colors.ListedColormap(["white", "black", "#a8d8ea"])


def _base_grid(world: GridWorld) -> np.ndarray:
    """Return a grid with obstacles marked and everything else free."""
    grid = np.zeros((world.rows, world.cols), dtype=int)
    for r, c in world.obstacles:
        grid[r, c] = 1
    return grid


def _setup_axes(ax, world: GridWorld) -> None:
    """Apply the shared grid styling used by every frame."""
    ax.set_xticks(np.arange(-0.5, world.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, world.rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(world.start[1], world.start[0], marker="s", color="green",
            markersize=14, zorder=4, label="Start")
    ax.plot(world.goal[1], world.goal[0], marker="*", color="gold",
            markersize=20, markeredgecolor="black", zorder=4, label="Goal")
    ax.legend(loc="upper right", fontsize=9, labelspacing=1.0, borderpad=0.8)


def _save_or_show(anim, fig, savepath: Optional[str], fps: int, show: bool) -> None:
    """Write the animation to a GIF, display it, or both."""
    if savepath:
        anim.save(savepath, writer=animation.PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_search(
    world: GridWorld,
    result: AStarResult,
    title: str = "A* Search",
    steps_per_frame: int = 2,
    path_draw_frames: int = 25,
    path_hold_frames: int = 20,
    fps: int = 20,
    show: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """Replay a graph search (A* or Dijkstra) cell by cell.

    The animation has three phases: the explored region grows in the order
    cells were popped from the open set, then the path is traced backward
    from goal to start the way _reconstruct_path walks its parent pointers,
    then the finished path is held before the loop repeats.

    Args:
        world: the grid environment.
        result: the search result to replay.
        title: plot title.
        steps_per_frame: cells revealed per frame. Higher is faster and
            produces a smaller GIF.
        path_draw_frames: frames spent tracing the path back from the goal.
        path_hold_frames: frames holding the completed path at the end.
        fps: frames per second when writing a GIF.
        show: open an interactive window.
        savepath: if given, write a GIF to this path.
    """
    history = result.frontier_history
    path = result.path or []
    reveal_frames = max(1, -(-len(history) // steps_per_frame))  # ceiling division
    draw_frames = path_draw_frames if path else 0
    total_frames = reveal_frames + draw_frames + path_hold_frames

    fig, ax = plt.subplots(figsize=(world.cols * 0.5, world.rows * 0.5))
    image = ax.imshow(_base_grid(world), cmap=CMAP, origin="upper", vmin=0, vmax=2)
    path_line, = ax.plot([], [], color="red", linewidth=2.5, zorder=3)
    _setup_axes(ax, world)

    def update(frame: int):
        # Phase 1: reveal the explored region
        revealed = min(len(history), (frame + 1) * steps_per_frame)
        grid = _base_grid(world)
        for r, c in history[:revealed]:
            if grid[r, c] == 0:
                grid[r, c] = 2
        image.set_data(grid)

        if frame < reveal_frames:
            path_line.set_data([], [])
            ax.set_title(f"{title}\nexplored: {revealed} of {len(history)}")
            return image, path_line

        # Phase 2: trace the path backward from the goal
        if frame < reveal_frames + draw_frames:
            progress = (frame - reveal_frames + 1) / draw_frames
            drawn = max(2, int(round(len(path) * progress)))
            segment = path[-drawn:]  # grows backward from the goal
            path_line.set_data([c[1] for c in segment], [c[0] for c in segment])
            ax.set_title(f"{title}\ntracing path back from goal")
            return image, path_line

        # Phase 3: hold the completed path
        path_line.set_data([c[1] for c in path], [c[0] for c in path])
        ax.set_title(f"{title}\npath length: {result.path_length}   "
                     f"explored: {len(result.explored)}")
        return image, path_line

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // fps, blit=False, repeat=True
    )
    _save_or_show(anim, fig, savepath, fps, show)


def animate_rrt(
    world: GridWorld,
    result: RRTResult,
    title: str = "RRT Search",
    edges_per_frame: int = 3,
    path_hold_frames: int = 20,
    fps: int = 20,
    show: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """Replay an RRT or RRT* search edge by edge.

    Args:
        world: the grid environment.
        result: the search result to replay.
        title: plot title.
        edges_per_frame: tree edges added per frame.
        path_hold_frames: extra frames held at the end showing the final path.
        fps: frames per second when writing a GIF.
        show: open an interactive window.
        savepath: if given, write a GIF to this path.
    """
    edges = result.tree_edges
    reveal_frames = max(1, -(-len(edges) // edges_per_frame))
    total_frames = reveal_frames + path_hold_frames

    fig, ax = plt.subplots(figsize=(world.cols * 0.5, world.rows * 0.5))
    image = ax.imshow(_base_grid(world), cmap=CMAP, origin="upper", vmin=0, vmax=2)
    tree_lines = LineCollectionProxy(ax)
    path_line, = ax.plot([], [], color="red", linewidth=2.5, zorder=3)
    _setup_axes(ax, world)

    def update(frame: int):
        revealed = min(len(edges), (frame + 1) * edges_per_frame)
        shown = edges[:revealed]

        grid = _base_grid(world)
        for parent, child in shown:
            for r, c in (parent, child):
                if grid[r, c] == 0:
                    grid[r, c] = 2
        image.set_data(grid)
        tree_lines.set_edges(shown)

        if frame >= reveal_frames - 1 and result.path:
            path_line.set_data([c[1] for c in result.path],
                               [c[0] for c in result.path])
            ax.set_title(f"{title}\npath length: {result.path_length}   "
                         f"nodes: {len(result.explored)}")
        else:
            path_line.set_data([], [])
            ax.set_title(f"{title}\nedges: {revealed} of {len(edges)}")

        return image, path_line

    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // fps, blit=False, repeat=True
    )
    _save_or_show(anim, fig, savepath, fps, show)


class LineCollectionProxy:
    """Manages the growing set of tree edges drawn on an axis.

    matplotlib has no cheap way to append to an existing line, so this keeps
    one Line2D per edge and reuses them across frames.
    """

    def __init__(self, ax):
        self._ax = ax
        self._lines: List = []

    def set_edges(self, edges: List[Tuple[Cell, Cell]]) -> None:
        # Extend the pool if this frame needs more lines than we have
        while len(self._lines) < len(edges):
            line, = self._ax.plot([], [], color="gray", linewidth=0.6,
                                  alpha=0.7, zorder=2)
            self._lines.append(line)

        for line, (parent, child) in zip(self._lines, edges):
            line.set_data([parent[1], child[1]], [parent[0], child[0]])

        # Blank any leftover lines from a longer previous frame
        for line in self._lines[len(edges):]:
            line.set_data([], [])