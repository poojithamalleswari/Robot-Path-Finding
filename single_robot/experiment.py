"""
experiment.py

Runs the heuristic comparison for the single-robot A* problem.

For every (grid_size, obstacle_density) configuration we generate a
batch of random solvable grids, and every heuristic is tested on the
*same* batch of grids -- otherwise a "lucky" easy grid could make a
weaker heuristic look better than it is. Results go to
results/single_robot_results.csv, and a few summary plots go to
results/*.png.
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from astar import astar
from grid_generator import make_instance
from heuristics import HEURISTICS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [20, 40, 60]
DENSITIES = [0.10, 0.20, 0.30]
TRIALS_PER_CONFIG = 30
MASTER_SEED = 42


def run_all():
    rng = random.Random(MASTER_SEED)
    rows = []

    for size in GRID_SIZES:
        for density in DENSITIES:
            for trial in range(TRIALS_PER_CONFIG):
                grid, start, goal = make_instance(size, density, rng)

                for name, h in HEURISTICS.items():
                    result = astar(grid, start, goal, h)
                    rows.append({
                        "grid_size": size,
                        "density": density,
                        "trial": trial,
                        "heuristic": name,
                        "found": result["found"],
                        "cost": result["cost"],
                        "nodes_expanded": result["nodes_expanded"],
                        "nodes_generated": result["nodes_generated"],
                        "runtime_ms": result["runtime_ms"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "single_robot_results.csv"), index=False)
    return df


def _optimality_ratio(df):
    """
    For each (size, density, trial) where every heuristic found a
    path, express each heuristic's cost as a ratio to the *shortest*
    cost found across heuristics on that instance (our stand-in for
    the true optimum, since manhattan/euclidean/chebyshev are all
    admissible and should all land on the optimum).
    """
    pivot = df.pivot_table(index=["grid_size", "density", "trial"],
                            columns="heuristic", values="cost")
    pivot = pivot.dropna()
    best = pivot.min(axis=1)
    ratio = pivot.div(best, axis=0)
    return ratio


def make_plots(df):
    heuristics = list(HEURISTICS.keys())
    colors = {"manhattan": "#2b6cb0", "euclidean": "#dd6b20",
              "chebyshev": "#38a169", "weighted_manhattan": "#c53030"}

    # 1. Nodes expanded vs obstacle density (averaged over grid size & trials)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in heuristics:
        sub = df[df.heuristic == name].groupby("density")["nodes_expanded"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Obstacle density")
    ax.set_ylabel("Mean nodes expanded")
    ax.set_title("Search effort vs. obstacle density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "nodes_vs_density.png"), dpi=150)
    plt.close(fig)

    # 2. Nodes expanded vs grid size
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in heuristics:
        sub = df[df.heuristic == name].groupby("grid_size")["nodes_expanded"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Grid size (N x N)")
    ax.set_ylabel("Mean nodes expanded")
    ax.set_title("Search effort vs. grid size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "nodes_vs_gridsize.png"), dpi=150)
    plt.close(fig)

    # 3. Runtime vs grid size
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in heuristics:
        sub = df[df.heuristic == name].groupby("grid_size")["runtime_ms"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=name, color=colors[name])
    ax.set_xlabel("Grid size (N x N)")
    ax.set_ylabel("Mean runtime (ms)")
    ax.set_title("Runtime vs. grid size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "runtime_vs_gridsize.png"), dpi=150)
    plt.close(fig)

    # 4. Path-cost optimality ratio (bar chart, 1.0 = optimal)
    ratio = _optimality_ratio(df)
    means = ratio.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(means.index, means.values,
                   color=[colors[n] for n in means.index])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Path cost / best found path cost")
    ax.set_title("Solution quality by heuristic (1.0 = optimal)")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=20)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                 ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "optimality_ratio.png"), dpi=150)
    plt.close(fig)

    # 5. Summary table
    summary = df.groupby("heuristic").agg(
        success_rate=("found", "mean"),
        mean_nodes_expanded=("nodes_expanded", "mean"),
        mean_nodes_generated=("nodes_generated", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
        mean_cost=("cost", "mean"),
    ).round(3)
    summary["mean_optimality_ratio"] = ratio.mean()
    summary.to_csv(os.path.join(RESULTS_DIR, "single_robot_summary.csv"))
    print(summary)
    return summary


def save_example_visual(size=25, density=0.22, seed=7):
    """One annotated example grid + path, used as a report screenshot."""
    grid, start, goal = make_instance(size, density, random.Random(seed))
    result = astar(grid, start, goal, HEURISTICS["manhattan"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
    if result["found"]:
        ys = [p[0] for p in result["path"]]
        xs = [p[1] for p in result["path"]]
        ax.plot(xs, ys, color="#2b6cb0", linewidth=2, label="Path")
    ax.scatter([start[1]], [start[0]], color="#38a169", s=120, marker="o", label="Start", zorder=5)
    ax.scatter([goal[1]], [goal[0]], color="#c53030", s=120, marker="*", label="Goal", zorder=5)
    ax.set_title(f"Single-robot A* example ({size}x{size}, density={density})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "example_grid_path.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = run_all()
    make_plots(df)
    save_example_visual()
    print("Done. Results in", RESULTS_DIR)
