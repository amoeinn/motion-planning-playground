"""A* pathfinding on a 2D grid.

Implements A* with configurable heuristic. Returns the shortest path from
start to goal, along with statistics about the search for analysis and
visualization.
"""

from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Callable, Dict, List, Optional, Set

from .grid_world import Cell, GridWorld


# A heuristic takes two cells and returns an estimated cost between them.
Heuristic = Callable[[Cell, Cell], float]


def manhattan(a: Cell, b: Cell) -> float:
    """Manhattan distance: sum of absolute row and column differences.

    Admissible for 4-connected grids with uniform step cost.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class AStarResult:
    """The outcome of an A* search."""

    path: Optional[List[Cell]]  # None if no path exists
    explored: Set[Cell] = field(default_factory=set)  # cells added to closed set
    frontier_history: List[Cell] = field(default_factory=list)  # cells popped, in order

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def path_length(self) -> Optional[int]:
        return len(self.path) - 1 if self.path else None


def astar(
    world: GridWorld,
    heuristic: Heuristic = manhattan,
) -> AStarResult:
    """Find the shortest path from world.start to world.goal using A*.

    Returns an AStarResult with the path (if found) and search statistics.
    Step cost is 1 per move (4-connected grid).
    """
    world.validate()

    # Priority queue: entries are (f_score, tiebreak_counter, cell)
    # The counter ensures stable ordering when f_scores are equal.
    open_heap: List = []
    counter = 0
    heappush(open_heap, (heuristic(world.start, world.goal), counter, world.start))

    # g_score[cell] = actual cost from start to reach cell
    g_score: Dict[Cell, float] = {world.start: 0.0}

    # came_from[cell] = the cell we arrived from on the best known path
    came_from: Dict[Cell, Cell] = {}

    # Cells whose neighbors have been evaluated
    closed: Set[Cell] = set()

    # Track the order we pop cells for visualization/analysis
    frontier_history: List[Cell] = []

    while open_heap:
        _, _, current = heappop(open_heap)

        # Skip stale entries: a cell can appear multiple times in the heap
        # if we found a shorter path to it after adding it the first time.
        if current in closed:
            continue

        closed.add(current)
        frontier_history.append(current)

        if current == world.goal:
            path = _reconstruct_path(came_from, current)
            return AStarResult(path=path, explored=closed, frontier_history=frontier_history)

        for neighbor in world.neighbors_4(current):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + 1.0  # step cost = 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, world.goal)
                counter += 1
                heappush(open_heap, (f_score, counter, neighbor))

    # Open set exhausted without reaching goal
    return AStarResult(path=None, explored=closed, frontier_history=frontier_history)


def _reconstruct_path(came_from: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
    """Walk backwards from goal using came_from pointers, then reverse."""
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def zero_heuristic(a: Cell, b: Cell) -> float:
    """Return 0 regardless of inputs. Used to reduce A* to Dijkstra."""
    return 0.0


def dijkstra(world: GridWorld) -> AStarResult:
    """Find the shortest path from world.start to world.goal using Dijkstra.

    Dijkstra is exactly A* with a zero heuristic: it explores by g-score only,
    with no bias toward the goal. This produces a uniform-cost search that
    expands outward from start in all directions until the goal is popped.

    Result is optimal (Dijkstra is the classic optimal shortest-path algorithm)
    but explores strictly more cells than A* with an informative heuristic.
    """
    return astar(world, heuristic=zero_heuristic)
