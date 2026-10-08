"""
cooperative_astar.py

Our proposed fix for the collisions produced by independent_astar.py:
prioritized planning over a space-time grid.

Idea
----
1. Decide a planning order for the robots (we plan the ones with the
   shortest individual path first -- this tends to leave more room
   for the harder, longer-range robots to route around them).
2. Plan robots one at a time. Each robot searches over states
   (position, time) instead of just position, using A* with the
   Manhattan distance to goal as the heuristic (time doesn't affect
   the heuristic, only g).
3. While planning robot k, its search may not:
     - occupy a cell at a time step already reserved by an earlier
       robot (vertex constraint), or
     - swap cells with an earlier robot between two consecutive time
       steps (edge constraint), or
     - move into a cell that an earlier robot has already parked in
       and never leaves again (that robot's goal cell after its own
       arrival time).
4. Once robot k's path is found, its cells (and its "parked at goal"
   tail) are added to the reservation table before planning robot
   k+1.

This is the classic "cooperative A*" / prioritized-planning approach.
It is not complete in theory (a bad priority order can make an
instance appear unsolvable even though a solution exists), but it
scales far better than trying to search the joint state space of all
robots at once, and in practice resolves the great majority of
instances that independent A* fails on.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from heuristics import manhattan

sys.path.insert(0, os.path.dirname(__file__))
from spacetime_astar import space_time_astar


def _shortest_path_len(grid, start, goal):
    """Plain BFS distance, used only to decide planning priority."""
    from collections import deque
    size = grid.shape[0]
    seen = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return seen[cur]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < size and 0 <= nxt[1] < size and grid[nxt] == 0 and nxt not in seen:
                seen[nxt] = seen[cur] + 1
                q.append(nxt)
    return float("inf")


def plan_cooperative(grid, starts, goals, horizon_multiplier=3, extra_buffer=10):
    t0 = time.perf_counter()
    n = len(starts)

    order = sorted(range(n), key=lambda i: _shortest_path_len(grid, starts[i], goals[i]))

    size = grid.shape[0]
    horizon = size * horizon_multiplier + extra_buffer

    reserved_vertices = set()
    reserved_edges = set()
    parked = {}  # goal_cell -> time from which it's permanently occupied

    paths = [None] * n
    success = True

    for idx in order:
        path = space_time_astar(grid, starts[idx], goals[idx],
                                  reserved_vertices, reserved_edges,
                                  parked, horizon)
        if path is None:
            success = False
            break

        paths[idx] = path
        for t, pos in enumerate(path):
            reserved_vertices.add((pos, t))
        for t in range(len(path) - 1):
            reserved_edges.add((path[t], path[t + 1], t))
        # this robot is permanently parked at its goal from arrival time on
        parked[path[-1]] = min(parked.get(path[-1], float("inf")), len(path) - 1)

    runtime_ms = (time.perf_counter() - t0) * 1000

    if not success:
        return {
            "success": False,
            "paths": None,
            "runtime_ms": runtime_ms,
            "sum_of_costs": None,
        }

    return {
        "success": True,
        "paths": paths,
        "runtime_ms": runtime_ms,
        "sum_of_costs": sum(len(p) - 1 for p in paths),
    }
