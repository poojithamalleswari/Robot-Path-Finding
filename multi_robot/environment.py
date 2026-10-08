"""
environment.py

Builds a shared grid with N robots, each with its own random start and
goal cell. Every start/goal pair is checked for reachability with a
plain BFS (ignoring other robots -- that's exactly the assumption that
breaks down once robots share the map, which is the point of the
multi-robot experiment).
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from grid_generator import _is_reachable, MOVES_4  # reuse the BFS check

import numpy as np


def make_multi_instance(size, obstacle_density, n_agents, rng=None, max_attempts=500):
    rng = rng or random.Random()

    for _ in range(max_attempts):
        grid = np.zeros((size, size), dtype=np.uint8)
        n_cells = size * size
        n_obstacles = int(n_cells * obstacle_density)
        obstacle_cells = rng.sample(range(n_cells), n_obstacles)
        for idx in obstacle_cells:
            grid[idx // size, idx % size] = 1

        free_cells = [(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0]
        needed = 2 * n_agents
        if len(free_cells) < needed:
            continue

        chosen = rng.sample(free_cells, needed)
        starts = chosen[:n_agents]
        goals = chosen[n_agents:]

        ok = True
        for s, g in zip(starts, goals):
            if not _is_reachable(grid, s, g):
                ok = False
                break
        if not ok:
            continue

        # also require starts to be distinct from each other and goals
        # distinct from each other (rng.sample already guarantees this
        # since we drew all 2*n_agents cells without replacement)
        return grid, starts, goals

    raise RuntimeError(
        f"Could not build a solvable {size}x{size} multi-robot instance "
        f"with {n_agents} agents at density {obstacle_density}"
    )
