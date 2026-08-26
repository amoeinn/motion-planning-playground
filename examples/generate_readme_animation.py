"""Generate the animated search GIF embedded in the README.

Replays A* on the wall world and writes docs/astar_search.gif. Uses the
Agg backend so no display is needed.

Usage:
    python examples/generate_readme_animation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed

from src.animate import animate_search
from src.astar import astar
from src.grid_world import make_wall_world


def main() -> None:
    world = make_wall_world(rows=15, cols=20)
    result = astar(world)

    output_path = Path(__file__).resolve().parent.parent / "docs" / "astar_search.gif"
    output_path.parent.mkdir(exist_ok=True)

    animate_search(
        world,
        result,
        title="A* on Wall World",
        steps_per_frame=2,
        path_draw_frames=25,
        path_hold_frames=25,
        fps=20,
        show=False,
        savepath=str(output_path),
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()