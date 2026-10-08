"""
astar.py

Plain A* search on a 4-connected occupancy grid. Kept deliberately
close to the textbook version (a binary heap open list, a closed set,
g/f tracking through a dict of Node objects) so it's easy to swap
heuristics in and out for the comparison in experiment.py.
"""

import heapq
import itertools
import time

MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Node:
    __slots__ = ("pos", "g", "h", "parent")

    def __init__(self, pos, g, h, parent):
        self.pos = pos
        self.g = g
        self.h = h
        self.parent = parent

    @property
    def f(self):
        return self.g + self.h


def _in_bounds(pos, size):
    r, c = pos
    return 0 <= r < size and 0 <= c < size


def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.pos)
        node = node.parent
    path.reverse()
    return path


def astar(grid, start, goal, heuristic, node_limit=200_000, tie_break="low_h"):
    """
    Run A* from start to goal on `grid` using `heuristic(pos, goal)`.

    Returns a dict with:
        found          -- bool
        path            -- list of (r, c), empty if not found
        cost            -- path length in moves (None if not found)
        nodes_expanded  -- number of nodes popped off the open list
        nodes_generated -- number of nodes pushed onto the open list
        runtime_ms      -- wall clock time for the search
    """
    size = grid.shape[0]
    t0 = time.perf_counter()

    # Tie-breaking among nodes with equal f matters a lot on grids, where
    # huge numbers of cells share the same f. Two deterministic rules:
    #   "low_h" (default): prefer the node with the smaller h, i.e. the one
    #                      closer to the goal -- the standard choice.
    #   "fifo":            prefer the node that was inserted first.
    # Both finish with an insertion counter so runs are fully reproducible.
    # (An earlier version tie-broke on the node's memory address, which made
    # results differ from run to run.)
    prefer_low_h = tie_break == "low_h"
    tie = itertools.count()
    start_node = Node(start, 0, heuristic(start, goal), None)
    open_heap = [(start_node.f, start_node.h if prefer_low_h else 0, next(tie), start_node)]
    best_g = {start: 0}

    nodes_expanded = 0
    nodes_generated = 1

    while open_heap:
        if nodes_expanded > node_limit:
            break

        _, _, _, current = heapq.heappop(open_heap)

        if current.g > best_g.get(current.pos, float("inf")):
            # stale heap entry
            continue

        nodes_expanded += 1

        if current.pos == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            path = reconstruct_path(current)
            return {
                "found": True,
                "path": path,
                "cost": current.g,
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "runtime_ms": runtime_ms,
            }

        for dr, dc in MOVES_4:
            nxt = (current.pos[0] + dr, current.pos[1] + dc)
            if not _in_bounds(nxt, size) or grid[nxt] == 1:
                continue

            tentative_g = current.g + 1
            if tentative_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = tentative_g
                h = heuristic(nxt, goal)
                new_node = Node(nxt, tentative_g, h, current)
                heapq.heappush(open_heap, (new_node.f, h if prefer_low_h else 0, next(tie), new_node))
                nodes_generated += 1

    runtime_ms = (time.perf_counter() - t0) * 1000
    return {
        "found": False,
        "path": [],
        "cost": None,
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "runtime_ms": runtime_ms,
    }
