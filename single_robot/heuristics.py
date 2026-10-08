"""
heuristics.py

Four heuristics for A* on a 4-connected grid (robot can move
up/down/left/right, each move costs 1).

manhattan   -- |dr| + |dc|. This is exactly the cost of an optimal
               path on an *open* grid, so it's the tightest admissible
               heuristic of the bunch here.
euclidean   -- straight-line distance. Admissible (never overestimates
               the true 4-connected cost) but looser than Manhattan.
chebyshev   -- max(|dr|, |dc|). Admissible here too (it's <= Manhattan),
               but the loosest of the three -- included mainly to show
               what happens when you under-inform the search.
weighted_manhattan -- 1.5 * Manhattan. NOT admissible, included on
               purpose as a "greedy" style heuristic so the report can
               show the classic speed-vs-optimality trade-off.
"""

import math


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def weighted_manhattan(a, b, w=1.5):
    return w * manhattan(a, b)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "weighted_manhattan": weighted_manhattan,
}

ADMISSIBLE = {
    "manhattan": True,
    "euclidean": True,
    "chebyshev": True,
    "weighted_manhattan": False,
}
