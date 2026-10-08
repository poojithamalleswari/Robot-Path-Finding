"""
analysis.py

Extra analysis on top of experiment.py's results:

1. Paired significance tests. Every heuristic in experiment.py was run
   on the *same* batch of grids (see experiment.py's docstring), which
   makes this a paired/matched-samples design -- exactly the setup a
   paired t-test or Wilcoxon signed-rank test is meant for, rather than
   an unpaired test that would throw away the pairing information.
   We run both for every pair of heuristics on nodes_expanded.

2. Distribution plots. The summary table only shows means; box plots
   show the spread and outliers that a single average can hide.

3. Node-expansion heatmaps. For one example grid, we instrument A* to
   record every cell it expands and render it as a heatmap per
   heuristic, so "Manhattan explores fewer cells" becomes a picture,
   not just a smaller number in a table.

Run this *after* experiment.py, since it reads single_robot_results.csv.
"""

import itertools
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from grid_generator import make_instance
from heuristics import HEURISTICS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_significance_tests():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "single_robot_results.csv"))

    # one row per (grid_size, density, trial) per heuristic -> pivot so
    # each heuristic becomes a column of matched observations
    pivot = df.pivot_table(index=["grid_size", "density", "trial"],
                            columns="heuristic", values="nodes_expanded")

    heuristics = list(HEURISTICS.keys())
    rows = []
    for h1, h2 in itertools.combinations(heuristics, 2):
        a, b = pivot[h1].values, pivot[h2].values
        t_stat, t_p = stats.ttest_rel(a, b)
        w_stat, w_p = stats.wilcoxon(a, b)
        rows.append({
            "heuristic_a": h1,
            "heuristic_b": h2,
            "mean_diff_nodes_expanded": a.mean() - b.mean(),
            "paired_t_statistic": t_stat,
            "paired_t_pvalue": t_p,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_p,
            "significant_at_0.001": bool(t_p < 0.001 and w_p < 0.001),
        })

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "significance_tests.csv"), index=False)
    print(sig_df.to_string(index=False))
    return sig_df


def make_distribution_plots():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "single_robot_results.csv"))
    heuristics = list(HEURISTICS.keys())
    colors = ["#2b6cb0", "#dd6b20", "#38a169", "#c53030"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    data_nodes = [df[df.heuristic == h]["nodes_expanded"].values for h in heuristics]
    bp = axes[0].boxplot(data_nodes, tick_labels=heuristics, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Nodes expanded")
    axes[0].set_title("Distribution of nodes expanded, by heuristic")
    axes[0].tick_params(axis="x", rotation=20)

    data_runtime = [df[df.heuristic == h]["runtime_ms"].values for h in heuristics]
    bp2 = axes[1].boxplot(data_runtime, tick_labels=heuristics, patch_artist=True, showfliers=True)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Runtime (ms)")
    axes[1].set_title("Distribution of runtime, by heuristic")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "distribution_boxplots.png"), dpi=150)
    plt.close(fig)


def _astar_with_visited(grid, start, goal, heuristic):
    """A small standalone copy of astar() (same tie-break rule: prefer lower h)
    that also returns every
    expanded cell, purely for the heatmap -- kept separate so the
    timed runs in experiment.py aren't slowed down by this bookkeeping."""
    import heapq

    size = grid.shape[0]
    MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    g0 = 0
    h0 = heuristic(start, goal)
    open_heap = [(g0 + h0, h0, 0, start, None)]
    best_g = {start: 0}
    visited_order = []
    counter = 1

    while open_heap:
        f, _, _, pos, parent = heapq.heappop(open_heap)
        if best_g.get(pos, float("inf")) < f - heuristic(pos, goal):
            continue
        visited_order.append(pos)
        if pos == goal:
            break
        for dr, dc in MOVES_4:
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if grid[nxt] == 1:
                continue
            tentative_g = best_g[pos] + 1
            if tentative_g < best_g.get(nxt, float("inf")):
                best_g[nxt] = tentative_g
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(nxt, goal), heuristic(nxt, goal), counter, nxt, pos))

    return visited_order


def make_heatmaps(size=40, density=0.22, seed=21):
    grid, start, goal = make_instance(size, density, random.Random(seed))
    heuristics = list(HEURISTICS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    counts = {}
    for ax, name in zip(axes.flat, heuristics):
        h = HEURISTICS[name]
        visited = _astar_with_visited(grid, start, goal, h)
        counts[name] = len(visited)

        heat = np.zeros_like(grid, dtype=float)
        for pos in visited:
            heat[pos] = 1
        heat[grid == 1] = np.nan  # keep obstacles visually distinct

        masked = np.ma.masked_invalid(heat)
        cmap = plt.cm.YlOrRd
        cmap.set_bad(color="black")
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
        ax.scatter([start[1]], [start[0]], color="#2b6cb0", s=60, marker="o", zorder=5)
        ax.scatter([goal[1]], [goal[0]], color="#2b6cb0", s=90, marker="*", zorder=5)
        ax.set_title(f"{name}\n({len(visited)} cells expanded)")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Which cells each heuristic explored on the same {size}x{size} instance", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(RESULTS_DIR, "node_expansion_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.Series(counts, name="cells_expanded").to_csv(os.path.join(RESULTS_DIR, "heatmap_counts.csv"))


if __name__ == "__main__":
    run_significance_tests()
    make_distribution_plots()
    make_heatmaps()
    print("Done. See", RESULTS_DIR)
