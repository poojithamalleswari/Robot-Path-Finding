"""
tiebreak_ablation.py

How much does the tie-breaking rule matter? On a 4-connected grid with
unit costs, a great many cells share exactly the same f = g + h, so
the order in which A* expands them is decided entirely by the
tie-break rule. Same 270 grids and seed as experiment.py; every
admissible heuristic is run under both rules and the mean nodes
expanded compared. Path cost is identical either way (both rules are
still A* with an admissible heuristic) -- only the effort changes.
"""
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from astar import astar
from experiment import DENSITIES, GRID_SIZES, MASTER_SEED, TRIALS_PER_CONFIG
from grid_generator import make_instance
from heuristics import HEURISTICS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    rng = random.Random(MASTER_SEED)
    rows = []
    for size in GRID_SIZES:
        for density in DENSITIES:
            for _ in range(TRIALS_PER_CONFIG):
                grid, start, goal = make_instance(size, density, rng)
                for name in ("manhattan", "euclidean", "chebyshev"):
                    for rule in ("low_h", "fifo"):
                        r = astar(grid, start, goal, HEURISTICS[name], tie_break=rule)
                        rows.append({"heuristic": name, "tie_break": rule,
                                     "nodes_expanded": r["nodes_expanded"], "cost": r["cost"]})
    df = pd.DataFrame(rows)
    out = df.pivot_table(index="heuristic", columns="tie_break", values="nodes_expanded", aggfunc="mean")
    out["fifo_over_low_h"] = out["fifo"] / out["low_h"]
    costs_equal = (df.pivot_table(index=df.index // 2, columns="tie_break", values="cost")
                   .eval("low_h == fifo").all())
    out = out.round(3)
    out.to_csv(os.path.join(RESULTS_DIR, "tiebreak_ablation.csv"))
    print(out)
    print("path costs identical under both rules:", bool(costs_equal))


if __name__ == "__main__":
    main()
