"""
grid_generator.py

Builds random grid worlds for the robot path-finding experiments.

A grid is a 2D numpy array of 0/1 values:
    0 -> free cell
    1 -> obstacle

We also pick a random start and goal cell. To keep the experiments
meaningful, a grid is only accepted if a path between start and goal
actually exists (checked with a plain BFS flood-fill). If not, we
just resample the obstacles / start / goal until we get a solvable
instance -- this mirrors what most people do by hand when they draw
a maze and then double check it isn't broken.
"""

import random
from collections import deque

import numpy as np

# 4-connected moves: up, down, left, right
MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(pos, size):
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def _is_reachable(grid, start, goal):
    """Plain BFS over free cells -- just a connectivity check."""
    size = grid.shape[0]
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for dr, dc in MOVES_4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if _in_bounds(nxt, size) and nxt not in seen and grid[nxt] == 0:
                seen.add(nxt)
                q.append(nxt)
    return False


def make_instance(size, obstacle_density, rng=None, max_attempts=200):
    """
    Create one random grid instance.

    size             -- grid is size x size
    obstacle_density -- fraction of cells (excluding start/goal) that
                         are blocked, roughly in [0, 0.4] is sane
    rng              -- an instance of random.Random for reproducibility

    Returns (grid, start, goal). Raises RuntimeError if it can't find
    a solvable layout within max_attempts tries (only happens at very
    high densities).
    """
    rng = rng or random.Random()

    for _ in range(max_attempts):
        grid = np.zeros((size, size), dtype=np.uint8)
        n_cells = size * size
        n_obstacles = int(n_cells * obstacle_density)

        obstacle_cells = rng.sample(range(n_cells), n_obstacles)
        for idx in obstacle_cells:
            grid[idx // size, idx % size] = 1

        free_cells = [(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0]
        if len(free_cells) < 2:
            continue

        start, goal = rng.sample(free_cells, 2)
        grid[start] = 0
        grid[goal] = 0

        if _is_reachable(grid, start, goal):
            return grid, start, goal

    raise RuntimeError(
        f"Could not build a solvable {size}x{size} grid at density "
        f"{obstacle_density} after {max_attempts} attempts"
    )


if __name__ == "__main__":
    g, s, t = make_instance(10, 0.2, random.Random(1))
    print(g)
    print("start", s, "goal", t)
