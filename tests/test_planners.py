"""Tests for the grid world and the four planners.

The properties asserted here are the ones that would be real bugs if
violated: optimality where it is guaranteed, reproducibility where a seed
is supplied, and the invariant that a returned path is actually walkable
one cell at a time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.astar import astar, dijkstra
from src.grid_world import GridWorld, make_wall_world
from src.rrt import rrt, rrt_star

# The optimal path length on the default wall world, verified by A*,
# which is provably optimal with an admissible heuristic.
WALL_WORLD_OPTIMUM = 27


@pytest.fixture
def world():
    return make_wall_world(rows=15, cols=20)


@pytest.fixture
def sealed_world():
    """A world where the goal is walled off in its own corner."""
    obstacles = {(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)}
    return GridWorld(rows=10, cols=10, start=(9, 9), goal=(0, 0),
                     obstacles=obstacles)


def assert_walkable(world: GridWorld, path) -> None:
    """Every step in a path must move exactly one cell into free space."""
    assert path[0] == world.start
    assert path[-1] == world.goal
    for current, following in zip(path, path[1:]):
        distance = abs(current[0] - following[0]) + abs(current[1] - following[1])
        assert distance == 1, f"step from {current} to {following} spans {distance} cells"
        assert world.is_free(following), f"path enters blocked cell {following}"


class TestGridWorld:
    def test_obstacles_are_not_free(self, world):
        blocked = next(iter(world.obstacles))
        assert not world.is_free(blocked)

    def test_out_of_bounds_is_not_free(self, world):
        assert not world.is_free((-1, 0))
        assert not world.is_free((world.rows, 0))

    def test_neighbors_exclude_obstacles_and_edges(self, world):
        corner = world.neighbors_4((0, 0))
        assert set(corner) <= {(0, 1), (1, 0)}

    def test_validate_rejects_goal_on_obstacle(self, world):
        world.goal = next(iter(world.obstacles))
        with pytest.raises(ValueError):
            world.validate()


class TestGraphSearch:
    def test_astar_finds_optimal_path(self, world):
        result = astar(world)
        assert result.found
        assert result.path_length == WALL_WORLD_OPTIMUM

    def test_dijkstra_finds_optimal_path(self, world):
        result = dijkstra(world)
        assert result.found
        assert result.path_length == WALL_WORLD_OPTIMUM

    def test_heuristic_reduces_exploration(self, world):
        """A* should touch fewer cells than Dijkstra. That is the point."""
        assert len(astar(world).explored) < len(dijkstra(world).explored)

    def test_astar_path_is_walkable(self, world):
        assert_walkable(world, astar(world).path)

    def test_dijkstra_path_is_walkable(self, world):
        assert_walkable(world, dijkstra(world).path)

    def test_frontier_history_matches_explored(self, world):
        """Every popped cell is in the closed set, with no duplicates."""
        result = astar(world)
        assert len(result.frontier_history) == len(set(result.frontier_history))
        assert set(result.frontier_history) == result.explored


class TestSamplingPlanners:
    def test_rrt_finds_a_path(self, world):
        result = rrt(world, seed=42)
        assert result.found

    def test_rrt_star_finds_a_path(self, world):
        result = rrt_star(world, seed=42, max_iterations=1000)
        assert result.found

    def test_rrt_path_is_walkable(self, world):
        assert_walkable(world, rrt(world, seed=42).path)

    def test_rrt_star_path_is_walkable(self, world):
        """Regression test: rewiring once produced edges spanning several cells."""
        result = rrt_star(world, seed=42, max_iterations=1000)
        assert_walkable(world, result.path)

    def test_no_planner_beats_the_optimum(self, world):
        """Sampling planners can be worse than optimal, never better."""
        assert rrt(world, seed=42).path_length >= WALL_WORLD_OPTIMUM
        assert rrt_star(world, seed=42,
                        max_iterations=1000).path_length >= WALL_WORLD_OPTIMUM

    def test_rrt_is_reproducible(self, world):
        assert rrt(world, seed=7).path == rrt(world, seed=7).path

    def test_rrt_star_is_reproducible(self, world):
        first = rrt_star(world, seed=7, max_iterations=500)
        second = rrt_star(world, seed=7, max_iterations=500)
        assert first.path == second.path

    def test_tree_has_one_edge_per_non_root_node(self, world):
        result = rrt(world, seed=42)
        assert len(result.tree_edges) == len(result.explored) - 1


class TestUnreachableGoal:
    def test_astar_reports_no_path(self, sealed_world):
        assert not astar(sealed_world).found

    def test_dijkstra_reports_no_path(self, sealed_world):
        assert not dijkstra(sealed_world).found

    def test_rrt_reports_no_path(self, sealed_world):
        assert not rrt(sealed_world, seed=42, max_iterations=500).found

    def test_rrt_star_reports_no_path(self, sealed_world):
        assert not rrt_star(sealed_world, seed=42, max_iterations=500).found