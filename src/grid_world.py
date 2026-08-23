"""2D grid world for motion planning experiments.

A GridWorld represents a rectangular grid where each cell is either free or
blocked (obstacle). The world has a designated start and goal position, and
provides utilities to check validity, get neighbors, and generate obstacle
patterns for testing.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple

# A cell position: (row, col). Row 0 is top, col 0 is left.
Cell = Tuple[int, int]


@dataclass
class GridWorld:
    """A 2D grid environment with obstacles, a start, and a goal.

    Coordinates use (row, col) throughout. Row 0 is top, col 0 is left.
    """

    rows: int
    cols: int
    start: Cell
    goal: Cell
    obstacles: Set[Cell] = field(default_factory=set)

    def in_bounds(self, cell: Cell) -> bool:
        """Return True if the cell is inside the grid boundaries."""
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell: Cell) -> bool:
        """Return True if the cell is inside the grid and not an obstacle."""
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors_4(self, cell: Cell) -> List[Cell]:
        """Return the 4-connected free neighbors of a cell (up, down, left, right)."""
        r, c = cell
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [n for n in candidates if self.is_free(n)]

    def validate(self) -> None:
        """Raise ValueError if the world is malformed.

        A world is malformed if the start or goal is out of bounds, or if
        either lands on an obstacle cell.
        """
        if not self.in_bounds(self.start):
            raise ValueError(f"start {self.start} out of bounds")
        if not self.in_bounds(self.goal):
            raise ValueError(f"goal {self.goal} out of bounds")
        if self.start in self.obstacles:
            raise ValueError(f"start {self.start} is on an obstacle")
        if self.goal in self.obstacles:
            raise ValueError(f"goal {self.goal} is on an obstacle")


def make_wall_world(rows: int = 15, cols: int = 20) -> GridWorld:
    """Create a demo world with a vertical wall obstacle and a gap.

    The wall runs down the middle with a gap near the top, forcing any planner
    to route around or through the gap.
    """
    wall_col = cols // 2
    gap_row = 2
    obstacles = {(r, wall_col) for r in range(rows) if r != gap_row}
    return GridWorld(
        rows=rows,
        cols=cols,
        start=(rows // 2, 1),
        goal=(rows // 2, cols - 2),
        obstacles=obstacles,
    )
