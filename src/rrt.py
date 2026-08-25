"""RRT (Rapidly-exploring Random Tree) pathfinding on a 2D grid.

RRT is a sampling-based planner: it builds a tree of collision-free
configurations by repeatedly sampling random points, finding the nearest
node in the tree, and extending toward the sample by a fixed step size.
Unlike A* and Dijkstra, RRT scales to continuous and high-dimensional
spaces where enumerating all states is infeasible.

This implementation operates on a 2D grid so it can be compared directly
against A* and Dijkstra on the same problem.
"""

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Set, Tuple

from .grid_world import Cell, GridWorld


@dataclass
class RRTResult:
    """The outcome of an RRT search."""

    path: Optional[List[Cell]]  # None if no path found within max iterations
    explored: Set[Cell] = field(default_factory=set)  # all node positions in the tree
    tree_edges: List[Tuple[Cell, Cell]] = field(default_factory=list)  # (parent, child)
    iterations_used: int = 0  # how many iterations before terminating

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def path_length(self) -> Optional[int]:
        return len(self.path) - 1 if self.path else None


def rrt(
    world: GridWorld,
    step_size: int = 1,
    goal_bias: float = 0.05,
    max_iterations: int = 5000,
    seed: Optional[int] = None,
) -> RRTResult:
    """Find a path from world.start to world.goal using RRT.

    Args:
        world: the grid environment.
        step_size: how many cells to extend per iteration (default 1).
        goal_bias: probability of sampling the goal directly instead of a
            random cell (default 0.05).
        max_iterations: give up after this many iterations if goal not reached.
        seed: random seed for reproducibility. None uses system entropy.

    Returns:
        RRTResult with path (if found), tree structure, and iteration count.
    """
    world.validate()
    rng = Random(seed)

    # Tree structure: parent[child] = parent, where each key is a node position
    # in the tree and the value is the node it was extended from. The start
    # has no parent, so it's not in this dict.
    parent: Dict[Cell, Cell] = {}
    nodes: List[Cell] = [world.start]
    tree_edges: List[Tuple[Cell, Cell]] = []

    for iteration in range(1, max_iterations + 1):
        # Sample: goal with probability goal_bias, else uniform random cell
        if rng.random() < goal_bias:
            sample = world.goal
        else:
            sample = (rng.randint(0, world.rows - 1), rng.randint(0, world.cols - 1))

        # Find nearest node in tree (brute force is fine at this scale)
        nearest = _nearest_node(nodes, sample)

        # Steer from nearest toward sample by step_size cells
        new_node = _steer(nearest, sample, step_size)

        # Skip if new node is out of bounds, on an obstacle, or already in tree
        if not world.is_free(new_node):
            continue
        if new_node in parent or new_node == world.start:
            continue

        # Add new node to tree
        parent[new_node] = nearest
        nodes.append(new_node)
        tree_edges.append((nearest, new_node))

        # Check if we reached the goal
        if new_node == world.goal:
            path = _reconstruct_path(parent, world.goal)
            return RRTResult(
                path=path,
                explored=set(nodes),
                tree_edges=tree_edges,
                iterations_used=iteration,
            )

    # Max iterations exceeded without reaching goal
    return RRTResult(
        path=None,
        explored=set(nodes),
        tree_edges=tree_edges,
        iterations_used=max_iterations,
    )


def _nearest_node(nodes: List[Cell], target: Cell) -> Cell:
    """Return the node in the list closest to target by Euclidean distance."""
    return min(nodes, key=lambda n: (n[0] - target[0]) ** 2 + (n[1] - target[1]) ** 2)


def _steer(from_node: Cell, toward: Cell, step_size: int) -> Cell:
    """Take one step of size step_size from from_node toward toward.

    On a 4-connected grid, we pick the cardinal direction (up/down/left/right)
    that most reduces the distance to the target. This keeps new nodes on grid
    cells and matches the movement model used by A* and Dijkstra.
    """
    dr = toward[0] - from_node[0]
    dc = toward[1] - from_node[1]

    # Move in whichever axis has the larger absolute difference
    if abs(dr) > abs(dc):
        step_r = step_size if dr > 0 else -step_size
        return (from_node[0] + step_r, from_node[1])
    else:
        step_c = step_size if dc > 0 else -step_size
        return (from_node[0], from_node[1] + step_c)


def _reconstruct_path(parent: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
    """Walk backwards from goal using parent pointers, then reverse."""
    path = [goal]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path