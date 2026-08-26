"""Run RRT* on the wall world and display the result.

RRT* extends RRT with two improvements:
  1. Choose the parent that gives the lowest cost-to-reach (not just nearest)
  2. Rewire nearby nodes if going through the new node gives them a shorter path

These make RRT* asymptotically optimal on continuous-space problems. On this
4-connected grid, path length is bounded by grid geometry, so RRT and RRT*
often produce paths of the same length; RRT*'s advantage is more visible in
continuous or higher-dimensional planning problems.

Usage:
    python examples/run_rrt_star_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid_world import make_wall_world
from src.rrt import rrt_star
from src.visualize import plot_rrt_result


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    result = rrt_star(world, seed=42, max_iterations=1000)

    if result.found:
        print(f"Path found. Length: {result.path_length}, "
              f"nodes: {len(result.explored)}, "
              f"iterations run: {result.iterations_used}")
    else:
        print(f"No path found after {result.iterations_used} iterations. "
              f"Tree has {len(result.explored)} nodes.")

    plot_rrt_result(world, result, title="RRT* on Wall World")


if __name__ == "__main__":
    main()