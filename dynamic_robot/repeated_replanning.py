"""
repeated_replanning.py

Everything so far assumes the robot can see the whole grid before it
takes a single step. Real robots usually can't -- they only know what's
within sensor range, and have to react when they roll up to an
obstacle they didn't know was there.

The standard algorithm for this problem is D* Lite (Koenig & Likhachev,
2002): it's an *incremental* search, meaning that when a new obstacle
is discovered it repairs the existing search tree instead of starting
over, which is what makes it fast enough to replan after every single
step in real time.

This module implements the same navigate-with-limited-visibility
problem D* Lite solves, but with a simpler replanning rule to keep the
implementation approachable: every time the robot's sensor reveals a
cell that invalidates its current plan, it just reruns plain A* from
its current position on its current (partial) knowledge of the map,
rather than incrementally repairing the old search. This is sometimes
called "repeated A*" or "replanning A*" in the literature. It's
strictly more work per replan than D* Lite does -- D* Lite's entire
point is to avoid re-deriving information it already has -- but it
produces the same sequence of *moves*, which is what the experiment
and the simulation in this folder are about. A natural next step noted
in the report is swapping this function's internals for true D* Lite
without changing anything else in the pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from astar import astar
from heuristics import manhattan


def _sensed_cells(pos, radius, size):
    r0, c0 = pos
    cells = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < size and 0 <= c < size and abs(dr) + abs(dc) <= radius:
                cells.append((r, c))
    return cells


def navigate(true_grid, start, goal, sensor_radius=2, step_limit=5000):
    """
    Simulates one robot driving from start to goal with only partial
    knowledge of true_grid, re-sensing and replanning as it goes.

    Returns a dict with:
      success          -- reached the goal?
      actual_path      -- list of cells the robot actually visited
      replans          -- number of times a new plan was computed
      distance_traveled -- len(actual_path) - 1
      known_grid        -- the robot's final belief map (for visualizing)
    """
    size = true_grid.shape[0]
    import numpy as np
    known_grid = np.zeros_like(true_grid)   # optimistic: assume free until sensed
    known_mask = np.zeros_like(true_grid, dtype=bool)

    def sense(pos):
        for cell in _sensed_cells(pos, sensor_radius, size):
            known_mask[cell] = True
            known_grid[cell] = true_grid[cell]

    pos = start
    actual_path = [pos]
    replans = 0
    sense(pos)

    plan = astar(known_grid, pos, goal, manhattan)
    replans += 1
    if not plan["found"]:
        return {"success": False, "actual_path": actual_path, "replans": replans,
                "distance_traveled": 0, "known_grid": known_grid}
    current_plan = plan["path"][1:]  # exclude current cell

    steps = 0
    while pos != goal and steps < step_limit:
        steps += 1

        if not current_plan:
            return {"success": False, "actual_path": actual_path, "replans": replans,
                    "distance_traveled": len(actual_path) - 1, "known_grid": known_grid}

        next_cell = current_plan[0]

        if known_mask[next_cell] and known_grid[next_cell] == 1:
            # shouldn't normally happen (we replan as soon as we see a
            # blockage), but guard anyway and force a replan
            current_plan = None
        else:
            pos = next_cell
            actual_path.append(pos)
            current_plan = current_plan[1:]
            sense(pos)

            # did sensing reveal something that invalidates the rest
            # of the current plan?
            plan_still_valid = all(
                not (known_mask[c] and known_grid[c] == 1) for c in current_plan
            )
            if plan_still_valid:
                continue

        replan = astar(known_grid, pos, goal, manhattan)
        replans += 1
        if not replan["found"]:
            return {"success": False, "actual_path": actual_path, "replans": replans,
                    "distance_traveled": len(actual_path) - 1, "known_grid": known_grid}
        current_plan = replan["path"][1:]

    return {
        "success": pos == goal,
        "actual_path": actual_path,
        "replans": replans,
        "distance_traveled": len(actual_path) - 1,
        "known_grid": known_grid,
    }
