# motion-planning-playground

Interactive visualizations of 2D motion planning algorithms. Explore how A*, Dijkstra, RRT, and RRT* solve grid-world problems with obstacles, and compare their performance quantitatively.

## Algorithms

- **A***: grid-based optimal shortest path with heuristic guidance
- **Dijkstra**: uniform-cost graph search
- **RRT**: sampling-based tree exploration for continuous spaces
- **RRT***: asymptotically optimal variant with rewiring

## Requirements

Python 3.10+ with matplotlib and numpy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Early stage. Algorithm implementations and visualization tooling under development.
