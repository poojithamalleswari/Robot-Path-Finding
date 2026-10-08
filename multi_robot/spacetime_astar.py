"""
spacetime_astar.py

The single-agent search both multi-robot planners build on: A* over
(position, time) states rather than just positions, with a `wait`
action available at every cell. This is pulled out into its own
module because cooperative_astar.py (prioritized planning) and cbs.py
(Conflict-Based Search) both need exactly the same low-level search --
they only differ in *how the constraint sets are produced* (a
reservation table built up robot-by-robot for prioritized planning,
vs. constraints added one at a time down a search tree for CBS).
"""

import heapq
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from heuristics import manhattan


def space_time_astar(grid, start, goal, reserved_vertices, reserved_edges,
                      parked, horizon, node_limit=100_000):
    """
    reserved_vertices: set of (pos, t) that may not be occupied
    reserved_edges:    set (posA, posB, t) meaning "do not move
                        posB->posA during [t, t+1]" (blocks the swap)
    parked:             dict pos -> earliest time from which that cell
                        is permanently occupied (another robot already
                        parked there)
    horizon:            max time step to search to

    Returns the path (list of positions, index = time step) or None.
    """
    size = grid.shape[0]

    def blocked(pos, t):
        if pos in parked and t >= parked[pos]:
            return True
        return (pos, t) in reserved_vertices

    start_state = (start, 0)
    h0 = manhattan(start, goal)
    open_heap = [(h0, 0, 0, start_state, None)]
    best_g = {start_state: 0}
    parents = {start_state: None}
    nodes_expanded = 0
    counter = 1

    while open_heap:
        if nodes_expanded > node_limit:
            return None
        f, _, g, state, parent = heapq.heappop(open_heap)
        pos, t = state

        if g > best_g.get(state, float("inf")):
            continue

        parents[state] = parent
        nodes_expanded += 1

        if pos == goal:
            path = []
            cur = state
            while cur is not None:
                path.append(cur[0])
                cur = parents.get(cur)
            path.reverse()
            return path

        if t >= horizon:
            continue

        candidates = [pos] + [(pos[0] + dr, pos[1] + dc)
                               for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for nxt in candidates:
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if grid[nxt] == 1:
                continue
            nt = t + 1
            if blocked(nxt, nt):
                continue
            if (nxt, pos, t) in reserved_edges:
                continue
            tentative_g = g + 1
            nstate = (nxt, nt)
            if tentative_g < best_g.get(nstate, float("inf")):
                best_g[nstate] = tentative_g
                h = manhattan(nxt, goal)
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h, counter, tentative_g, nstate, state))

    return None
