"""
independent_astar.py

The "obvious first attempt" at multi-robot path-finding: just run
plain single-robot A* for every robot, completely ignoring the fact
that the other robots exist. Fast, but nothing stops two robots from
being scheduled through the same cell at the same time, or from
swapping places head-on -- both are counted as collisions below.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from astar import astar
from heuristics import manhattan


def plan_independent(grid, starts, goals):
    t0 = time.perf_counter()
    paths = []
    total_nodes = 0
    for s, g in zip(starts, goals):
        result = astar(grid, s, g, manhattan)
        total_nodes += result["nodes_expanded"]
        # a robot that has reached its goal is assumed to sit there and wait
        paths.append(result["path"])
    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "paths": paths,
        "runtime_ms": runtime_ms,
        "total_nodes_expanded": total_nodes,
        "sum_of_costs": sum(len(p) - 1 for p in paths),
    }


def _pad(paths):
    max_len = max(len(p) for p in paths)
    return [p + [p[-1]] * (max_len - len(p)) for p in paths]


def count_collisions(paths):
    """
    Returns (vertex_collision_events, edge_collision_events).

    vertex: two or more robots occupy the same cell at the same
            timestep.
    edge:   two robots swap cells between t and t+1 (a head-on crash
            you wouldn't catch by only checking vertices).
    """
    padded = _pad(paths)
    n_agents = len(padded)
    max_len = len(padded[0])

    vertex_events = 0
    for t in range(max_len):
        occupants = {}
        for i in range(n_agents):
            pos = padded[i][t]
            occupants.setdefault(pos, []).append(i)
        for pos, agents in occupants.items():
            if len(agents) > 1:
                vertex_events += len(agents) - 1

    edge_events = 0
    for t in range(max_len - 1):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if padded[i][t] == padded[j][t + 1] and padded[i][t + 1] == padded[j][t] \
                        and padded[i][t] != padded[i][t + 1]:
                    edge_events += 1

    return vertex_events, edge_events
