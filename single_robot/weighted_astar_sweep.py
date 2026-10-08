"""
weighted_astar_sweep.py

Formalizes the "weighted Manhattan" idea from heuristics.py into a
proper bounded-suboptimal A* study. Weighted A* uses h'(n) = w * h(n)
for some w >= 1; w = 1 is plain (optimal) A*, and larger w makes the
search greedier -- it commits harder to cells that look close to the
goal, generally expanding far fewer nodes at the cost of occasionally
returning a longer-than-optimal path.

A classical guarantee applies here: if h is admissible, weighted A*
with weight w is guaranteed to return a path no more than w times the
optimal cost (an "epsilon-admissible" or "bounded suboptimal" search).
This script sweeps w across a range and plots the resulting trade-off
curve (nodes expanded vs. solution quality), so the shape of that
trade-off is visible directly instead of just one fixed data point.
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from astar import astar
from grid_generator import make_instance
from heuristics import manhattan

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WEIGHTS = [1.0, 1.1, 1.2, 1.5, 1.75, 2.0, 2.5, 3.0]
GRID_SIZES = [30, 50]
DENSITIES = [0.15, 0.25]
TRIALS_PER_CONFIG = 25
MASTER_SEED = 7


def make_weighted_heuristic(w):
    def h(a, b):
        return w * manhattan(a, b)
    h.__name__ = f"weighted_{w}"
    return h


def run_sweep():
    rng = random.Random(MASTER_SEED)
    rows = []

    for size in GRID_SIZES:
        for density in DENSITIES:
            for trial in range(TRIALS_PER_CONFIG):
                grid, start, goal = make_instance(size, density, rng)

                for w in WEIGHTS:
                    h = make_weighted_heuristic(w)
                    result = astar(grid, start, goal, h)
                    rows.append({
                        "grid_size": size,
                        "density": density,
                        "trial": trial,
                        "weight": w,
                        "found": result["found"],
                        "cost": result["cost"],
                        "nodes_expanded": result["nodes_expanded"],
                        "runtime_ms": result["runtime_ms"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "weighted_astar_results.csv"), index=False)
    return df


def make_plots(df):
    pivot = df.pivot_table(index=["grid_size", "density", "trial"],
                            columns="weight", values="cost").dropna()
    optimal = pivot[1.0]  # w=1.0 is plain A*, always optimal
    ratio = pivot.div(optimal, axis=0)

    mean_ratio = ratio.mean()
    mean_nodes = df.groupby("weight")["nodes_expanded"].mean()
    mean_runtime = df.groupby("weight")["runtime_ms"].mean()

    summary = pd.DataFrame({
        "mean_nodes_expanded": mean_nodes,
        "mean_runtime_ms": mean_runtime,
        "mean_optimality_ratio": mean_ratio,
    }).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, "weighted_astar_summary.csv"))
    print(summary)

    # Trade-off curve: nodes expanded (search effort) vs solution quality
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(mean_nodes.values, mean_ratio.values, marker="o", color="#805ad5")
    for w, x, y in zip(mean_nodes.index, mean_nodes.values, mean_ratio.values):
        ax.annotate(f"w={w}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean nodes expanded (search effort)")
    ax.set_ylabel("Mean path cost / optimal path cost")
    ax.set_title("Weighted A*: speed vs. solution-quality trade-off")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "weighted_astar_tradeoff.png"), dpi=150)
    plt.close(fig)

    # Nodes expanded vs weight (effort drops sharply then flattens)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_nodes.index, mean_nodes.values, marker="o", color="#2b6cb0")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Mean nodes expanded")
    ax.set_title("Search effort vs. heuristic weight")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "weighted_astar_nodes_vs_weight.png"), dpi=150)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    df = run_sweep()
    make_plots(df)
    print("Done.")
