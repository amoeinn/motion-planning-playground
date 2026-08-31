# motion-planning-playground

Interactive visualizations of 2D motion planning algorithms. Explore how A*, Dijkstra, RRT, and RRT* solve grid-world problems with obstacles, and compare their performance quantitatively.

![A* vs Dijkstra vs RRT vs RRT* comparison](docs/comparison.png)

All four planners find a 27-step path around the wall, which is optimal for a 4-connected grid with unit-cost moves. A* explores 157 cells using its heuristic to focus toward the goal. Dijkstra explores 249 with no heuristic. RRT stops at 111 nodes once it first reaches the goal. RRT* runs the full 1000 iterations and rewires as it goes, producing a denser tree and a visibly straighter route; its asymptotic optimality shows up as path smoothness here rather than shorter length, since grid geometry already bounds all four at 27.

### Watching the search

![A* search animation](docs/astar_search.gif)

A* expands cells in order of estimated total cost, so the explored region grows toward the goal rather than uniformly outward. Once the goal is popped, the path is reconstructed by following parent pointers backward from the goal to the start, which is the direction the trace runs.

## Algorithms

- **A***: grid-based optimal shortest path with heuristic guidance (implemented)
- **Dijkstra**: uniform-cost graph search, implemented as A* with a zero heuristic (implemented)
- **RRT**: sampling-based tree exploration for continuous spaces (implemented)
- **RRT***: asymptotically optimal variant with rewiring (implemented)

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

Compare A*, Dijkstra, RRT, and RRT* on the same world:

```bash
python examples/run_comparison_demo.py
```

Run RRT alone on the wall world:

```bash
python examples/run_rrt_demo.py
```

Run RRT* alone on the wall world:

```bash
python examples/run_rrt_star_demo.py
```

Animate a search, replaying it step by step:

```bash
python examples/run_animation_demo.py            # A* (default)
python examples/run_animation_demo.py dijkstra
python examples/run_animation_demo.py rrt
python examples/run_animation_demo.py rrt_star
```

Regenerate the README image and animation:

```bash
python examples/generate_readme_image.py
python examples/generate_readme_animation.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The suite checks the properties that would be real bugs if violated: A* and Dijkstra return the optimal 27-step path, A* explores fewer cells than Dijkstra, every returned path moves one free cell at a time, seeded runs are reproducible, and all four planners report failure rather than hanging when the goal is walled off.

## Structure

```text
src/
  grid_world.py       2D grid environment with obstacles, start, goal
  astar.py            A* and Dijkstra implementations
  rrt.py              RRT and RRT* implementations
  visualize.py        matplotlib rendering
  animate.py          animated replay of a search
examples/
  run_astar_demo.py             A* on the wall world
  run_rrt_demo.py               RRT on the wall world
  run_rrt_star_demo.py          RRT* on the wall world
  run_comparison_demo.py        all four planners side by side
  run_animation_demo.py         animated replay, planner as argument
  generate_readme_image.py      docs image generator
  generate_readme_animation.py  docs animation generator
tests/
  test_planners.py              grid world and planner tests
```

## License

MIT, see [LICENSE](LICENSE).