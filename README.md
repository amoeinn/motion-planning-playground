# motion-planning-playground

Interactive visualizations of 2D motion planning algorithms. Explore how A*, Dijkstra, RRT, and RRT* solve grid-world problems with obstacles, and compare their performance quantitatively.

![A* vs Dijkstra comparison](docs/comparison.png)

Both algorithms find the optimal 27-step path around the wall, but A* explores 157 cells while Dijkstra explores 249. The heuristic doesn't change what A* finds, it changes how efficiently it finds it.

## Algorithms

- **A***: grid-based optimal shortest path with heuristic guidance (implemented)
- **Dijkstra**: uniform-cost graph search, implemented as A* with a zero heuristic (implemented)
- **RRT**: sampling-based tree exploration for continuous spaces (planned)
- **RRT***: asymptotically optimal variant with rewiring (planned)

## Requirements

Python 3.10+ with matplotlib and numpy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Examples

Run A* on the default wall world:

```bash
python examples/run_astar_demo.py
```

Compare A* against Dijkstra on the same world:

```bash
python examples/run_comparison_demo.py
```

Regenerate the README image:

```bash
python examples/generate_readme_image.py
```

## Structure

```text
src/
  grid_world.py       2D grid environment with obstacles, start, goal
  astar.py            A* and Dijkstra implementations
  visualize.py        matplotlib rendering
examples/
  run_astar_demo.py             single-algorithm demo
  run_comparison_demo.py        side-by-side A* vs Dijkstra
  generate_readme_image.py      docs image generator
```
