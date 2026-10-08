"""
cbs.py

Conflict-Based Search (CBS), after Sharon, Stern, Felner & Sturtevant,
"Conflict-Based Search for Optimal Multi-Agent Path Finding" (AAAI
2012). Unlike cooperative_astar.py's prioritized planner -- which
commits to a fixed planning order and can occasionally fail to find a
solution that does exist -- CBS is provably optimal (minimum
sum-of-costs) and complete, at the cost of potentially having to
explore many more branches when robots are packed into a tight space.
We use it in experiment.py as the "ground truth" to measure exactly
how suboptimal the faster prioritized planner's solutions are.

How it works
------------
CBS is a two-level search:

- High level: a best-first search over a *Constraint Tree* (CT). Each
  CT node carries one extra constraint beyond its parent's, plus a
  full solution (one path per agent) that is consistent with all of
  its constraints. We always expand the lowest-cost (sum-of-costs) CT
  node first.
- Low level: whenever a CT node needs an agent's path recomputed
  (because a new constraint was just added for that agent), a normal
  single-agent space-time A* search is run for that agent only, using
  the shared `space_time_astar` from spacetime_astar.py.

At the root, every agent is planned completely independently (no
constraints at all) -- identical to independent_astar.py. We then scan
the resulting set of paths for the first conflict. If there isn't one,
we're done and the solution is optimal. If there is one, we branch:
two children are created, each forbidding one of the two conflicting
agents from the offending (cell, time) [or edge], and only that one
agent's path is recomputed in each child. Both children are pushed
back onto the frontier and the search continues.

One deliberate simplification, scoped for this project: when a robot
drives into a cell permanently occupied by a robot that has *already
finished* at its goal there, we don't bother branching on the finished
robot (it has nowhere else to usefully go -- its goal is fixed, and
any branch that moves it away from its own goal can only cost more).
We instead add a single "stay-away-from-here-from-now-on" constraint
to the moving robot. This keeps the branching factor down without
giving up optimality for that class of conflict.
"""

import heapq
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from spacetime_astar import space_time_astar


def _pad(paths):
    max_len = max(len(p) for p in paths)
    return [p + [p[-1]] * (max_len - len(p)) for p in paths], max_len


def _find_first_conflict(paths):
    """
    Returns a conflict dict, or None if the solution is conflict-free.
    A conflict is either:
      {"type": "vertex", "a": i, "b": j, "pos": pos, "t": t,
       "a_parked": bool, "b_parked": bool}
      {"type": "edge", "a": i, "b": j, "pos_a": posA, "pos_b": posB, "t": t}
    """
    padded, max_len = _pad(paths)
    n = len(padded)

    for t in range(max_len):
        occupants = {}
        for i in range(n):
            occupants.setdefault(padded[i][t], []).append(i)
        for pos, agents in occupants.items():
            if len(agents) > 1:
                i, j = agents[0], agents[1]
                return {
                    "type": "vertex", "a": i, "b": j, "pos": pos, "t": t,
                    "a_parked": t >= len(paths[i]) - 1,
                    "b_parked": t >= len(paths[j]) - 1,
                }

    for t in range(max_len - 1):
        for i, j in itertools.combinations(range(n), 2):
            if padded[i][t] == padded[j][t + 1] and padded[i][t + 1] == padded[j][t] \
                    and padded[i][t] != padded[i][t + 1]:
                return {
                    "type": "edge", "a": i, "b": j,
                    "pos_a": padded[i][t], "pos_b": padded[i][t + 1], "t": t,
                }

    return None


class CTNode:
    __slots__ = ("vertex_constraints", "edge_constraints", "range_constraints",
                 "solution", "cost")

    def __init__(self, vertex_constraints, edge_constraints, range_constraints, solution, cost):
        self.vertex_constraints = vertex_constraints   # agent -> set of (pos, t)
        self.edge_constraints = edge_constraints       # agent -> set of (posA, posB, t)
        self.range_constraints = range_constraints     # agent -> dict pos -> min_t
        self.solution = solution                       # agent -> path
        self.cost = cost


def _replan_agent(grid, starts, goals, node, agent, horizon, node_limit):
    reserved_vertices = node.vertex_constraints[agent]
    # space_time_astar blocks move pos->nxt at time t when (nxt, pos, t)
    # is in reserved_edges -- see spacetime_astar.py's docstring.
    reserved_edges = {(b, a, t) for (a, b, t) in node.edge_constraints[agent]}
    parked = node.range_constraints[agent]
    return space_time_astar(grid, starts[agent], goals[agent],
                             reserved_vertices, reserved_edges, parked,
                             horizon, node_limit)


def solve_cbs(grid, starts, goals, horizon_multiplier=3, extra_buffer=10,
              ct_node_limit=1500, low_level_node_limit=20_000, time_limit_s=5.0):
    t0 = time.perf_counter()
    n = len(starts)
    size = grid.shape[0]
    horizon = size * horizon_multiplier + extra_buffer

    root_vertex = {i: set() for i in range(n)}
    root_edge = {i: set() for i in range(n)}
    root_range = {i: {} for i in range(n)}
    root_solution = {}
    for i in range(n):
        path = space_time_astar(grid, starts[i], goals[i], set(), set(), {},
                                 horizon, low_level_node_limit)
        if path is None:
            return {"success": False, "paths": None, "runtime_ms": (time.perf_counter() - t0) * 1000,
                    "sum_of_costs": None, "ct_nodes_expanded": 0, "reason": "root_agent_unreachable"}
        root_solution[i] = path

    root_cost = sum(len(p) - 1 for p in root_solution.values())
    root = CTNode(root_vertex, root_edge, root_range, root_solution, root_cost)

    counter = itertools.count()
    open_heap = [(root.cost, next(counter), root)]
    ct_nodes_expanded = 0

    while open_heap:
        if ct_nodes_expanded >= ct_node_limit or (time.perf_counter() - t0) > time_limit_s:
            return {"success": False, "paths": None, "runtime_ms": (time.perf_counter() - t0) * 1000,
                    "sum_of_costs": None, "ct_nodes_expanded": ct_nodes_expanded, "reason": "budget_exceeded"}

        _, _, node = heapq.heappop(open_heap)
        ct_nodes_expanded += 1

        paths_list = [node.solution[i] for i in range(n)]
        conflict = _find_first_conflict(paths_list)

        if conflict is None:
            return {
                "success": True,
                "paths": paths_list,
                "runtime_ms": (time.perf_counter() - t0) * 1000,
                "sum_of_costs": node.cost,
                "ct_nodes_expanded": ct_nodes_expanded,
                "reason": None,
            }

        if conflict["type"] == "vertex" and (conflict["a_parked"] != conflict["b_parked"]):
            # One agent is already finished and permanently parked --
            # only the still-moving agent needs a new constraint.
            moving = conflict["b"] if conflict["a_parked"] else conflict["a"]
            pos, t = conflict["pos"], conflict["t"]

            new_range = {k: dict(v) for k, v in node.range_constraints.items()}
            new_range[moving][pos] = min(new_range[moving].get(pos, t), t)

            child = CTNode(
                {k: set(v) for k, v in node.vertex_constraints.items()},
                {k: set(v) for k, v in node.edge_constraints.items()},
                new_range,
                dict(node.solution), node.cost,
            )
            new_path = _replan_agent(grid, starts, goals, child, moving, horizon, low_level_node_limit)
            if new_path is not None:
                child.solution[moving] = new_path
                child.cost = sum(len(p) - 1 for p in child.solution.values())
                heapq.heappush(open_heap, (child.cost, next(counter), child))
            continue

        # Standard binary branch: two children, one constraint each
        agents_to_branch = []
        if conflict["type"] == "vertex":
            pos, t = conflict["pos"], conflict["t"]
            agents_to_branch = [
                (conflict["a"], ("vertex", pos, t)),
                (conflict["b"], ("vertex", pos, t)),
            ]
        else:
            pos_a, pos_b, t = conflict["pos_a"], conflict["pos_b"], conflict["t"]
            agents_to_branch = [
                (conflict["a"], ("edge", pos_a, pos_b, t)),
                (conflict["b"], ("edge", pos_b, pos_a, t)),
            ]

        for agent, constraint in agents_to_branch:
            new_vertex = {k: set(v) for k, v in node.vertex_constraints.items()}
            new_edge = {k: set(v) for k, v in node.edge_constraints.items()}
            new_range = {k: dict(v) for k, v in node.range_constraints.items()}

            if constraint[0] == "vertex":
                _, pos, t = constraint
                new_vertex[agent].add((pos, t))
            else:
                _, pos_from, pos_to, t = constraint
                new_edge[agent].add((pos_from, pos_to, t))

            child = CTNode(new_vertex, new_edge, new_range, dict(node.solution), node.cost)
            new_path = _replan_agent(grid, starts, goals, child, agent, horizon, low_level_node_limit)
            if new_path is not None:
                child.solution[agent] = new_path
                child.cost = sum(len(p) - 1 for p in child.solution.values())
                heapq.heappush(open_heap, (child.cost, next(counter), child))

    return {"success": False, "paths": None, "runtime_ms": (time.perf_counter() - t0) * 1000,
            "sum_of_costs": None, "ct_nodes_expanded": ct_nodes_expanded, "reason": "exhausted"}
