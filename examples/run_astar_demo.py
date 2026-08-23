"""Run A* on the wall world and display the result.

Usage:
    python examples/run_astar_demo.py

From the repo root, with the venv activated.
"""

import sys
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.astar import astar
from src.grid_world import make_wall_world
from src.visualize import plot_result


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    result = astar(world)

    if result.found:
        print(f"Path found. Length: {result.path_length}, "
              f"explored: {len(result.explored)} cells.")
    else:
        print(f"No path found. Explored {len(result.explored)} cells.")

    plot_result(world, result, title="A* on Wall World")


if __name__ == "__main__":
    main()
