"""Animate a planner search on the wall world.

Watch the search unfold rather than only seeing its end state. A* and
Dijkstra replay cell by cell in the order they were popped from the open
set; RRT and RRT* replay edge by edge as the tree grows.

Usage:
    python examples/run_animation_demo.py            # A* (default)
    python examples/run_animation_demo.py dijkstra
    python examples/run_animation_demo.py rrt
    python examples/run_animation_demo.py rrt_star
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.animate import animate_rrt, animate_search
from src.astar import astar, dijkstra
from src.grid_world import make_wall_world
from src.rrt import rrt, rrt_star

PLANNERS = ("astar", "dijkstra", "rrt", "rrt_star")


def main() -> None:
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "astar"
    if choice not in PLANNERS:
        print(f"Unknown planner: {choice}. Choose one of: {', '.join(PLANNERS)}")
        raise SystemExit(1)

    world = make_wall_world(rows=15, cols=20)

    if choice == "astar":
        animate_search(world, astar(world), title="A* on Wall World")
    elif choice == "dijkstra":
        animate_search(world, dijkstra(world), title="Dijkstra on Wall World")
    elif choice == "rrt":
        animate_rrt(world, rrt(world, seed=42), title="RRT on Wall World")
    else:
        animate_rrt(world, rrt_star(world, seed=42, max_iterations=1000),
                    title="RRT* on Wall World")


if __name__ == "__main__":
    main()