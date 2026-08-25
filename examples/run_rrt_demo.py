"""Run RRT on the wall world and display the result.

Usage:
    python examples/run_rrt_demo.py

From the repo root, with the venv activated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grid_world import make_wall_world
from src.rrt import rrt
from src.visualize import plot_rrt_result


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    result = rrt(world, seed=42)

    if result.found:
        print(f"Path found in {result.iterations_used} iterations. "
              f"Length: {result.path_length}, nodes: {len(result.explored)}.")
    else:
        print(f"No path found after {result.iterations_used} iterations. "
              f"Tree has {len(result.explored)} nodes.")

    plot_rrt_result(world, result, title="RRT on Wall World")


if __name__ == "__main__":
    main()