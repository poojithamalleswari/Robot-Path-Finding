"""
cbs_experiment.py

experiment.py already shows independent A* vs. cooperative A* across
2-10 robots. This script adds the third leg of the comparison: CBS,
the optimal (minimum sum-of-costs) multi-robot planner, used here as
a ground truth to answer "how far from optimal is the fast prioritized
planner, really?"

CBS does not scale as gracefully as prioritized planning -- that's
the whole reason prioritized planning gets used in practice -- so this
sweep intentionally stays smaller (fewer robots, smaller grids) than
experiment.py's, and gives CBS a time/node budget per instance. When
CBS can't finish in budget we record that honestly rather than
pretending it solved the instance.

Metrics recorded, per the standard terminology used in the MAPF
literature (success rate, sum-of-costs, makespan, and solution
quality measured against a lower bound):
  - success rate for CBS itself (how often it finishes in budget)
  - CT (constraint-tree) nodes expanded -- CBS's own search-effort metric
  - suboptimality ratio: cooperative A*'s sum-of-costs divided by
    CBS's optimal sum-of-costs, and the same for makespan
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from environment import make_multi_instance
from cooperative_astar import plan_cooperative
from cbs import solve_cbs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [15, 20]
DENSITIES = [0.10, 0.20]
AGENT_COUNTS = [2, 4, 6, 8, 10, 12]
TRIALS_PER_CONFIG = 8
MASTER_SEED = 2024


def _makespan(paths):
    return max(len(p) - 1 for p in paths)


def run_all():
    rng = random.Random(MASTER_SEED)
    rows = []

    for size in GRID_SIZES:
        for density in DENSITIES:
            for n_agents in AGENT_COUNTS:
                for trial in range(TRIALS_PER_CONFIG):
                    try:
                        grid, starts, goals = make_multi_instance(size, density, n_agents, rng)
                    except RuntimeError:
                        continue

                    coop = plan_cooperative(grid, starts, goals)
                    cbs = solve_cbs(grid, starts, goals, time_limit_s=3.0, ct_node_limit=1500)

                    coop_makespan = _makespan(coop["paths"]) if coop["success"] else None
                    cbs_makespan = _makespan(cbs["paths"]) if cbs["success"] else None

                    rows.append({
                        "grid_size": size,
                        "density": density,
                        "n_agents": n_agents,
                        "trial": trial,
                        "coop_success": coop["success"],
                        "coop_sum_of_costs": coop["sum_of_costs"],
                        "coop_makespan": coop_makespan,
                        "coop_runtime_ms": coop["runtime_ms"],
                        "cbs_success": cbs["success"],
                        "cbs_sum_of_costs": cbs["sum_of_costs"],
                        "cbs_makespan": cbs_makespan,
                        "cbs_runtime_ms": cbs["runtime_ms"],
                        "cbs_ct_nodes_expanded": cbs["ct_nodes_expanded"],
                        "cbs_fail_reason": cbs.get("reason"),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "cbs_vs_cooperative_results.csv"), index=False)
    return df


def make_plots(df):
    both_ok = df[df.coop_success & df.cbs_success].copy()
    both_ok["soc_suboptimality"] = both_ok["coop_sum_of_costs"] / both_ok["cbs_sum_of_costs"]
    both_ok["makespan_suboptimality"] = both_ok["coop_makespan"] / both_ok["cbs_makespan"]

    # 1. CBS success rate vs agent count (its scalability limit)
    cbs_rate = df.groupby("n_agents")["cbs_success"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cbs_rate.index, cbs_rate.values, marker="o", color="#805ad5")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Fraction solved within budget")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("CBS success rate within a 3s / 1500-node budget")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cbs_success_rate.png"), dpi=150)
    plt.close(fig)

    # 2. Sum-of-costs suboptimality of cooperative A* relative to CBS-optimal
    subopt = both_ok.groupby("n_agents")["soc_suboptimality"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(subopt.index.astype(str), (subopt.values - 1) * 100, color="#2b6cb0")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Cooperative A* sum-of-costs above optimal (%)")
    ax.set_title("How far from optimal is prioritized planning?")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "coop_suboptimality_vs_agents.png"), dpi=150)
    plt.close(fig)

    # 3. CT nodes expanded vs agent count (CBS's own cost of being optimal)
    ct_nodes = df[df.cbs_success].groupby("n_agents")["cbs_ct_nodes_expanded"].mean()
    makespan_subopt = both_ok.groupby("n_agents")["makespan_suboptimality"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ct_nodes.index, ct_nodes.values, marker="o", color="#c53030")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Mean CT nodes expanded (CBS)")
    ax.set_title("CBS search effort grows with robot count")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cbs_ct_nodes_vs_agents.png"), dpi=150)
    plt.close(fig)

    # 4. Runtime: cooperative A* vs CBS
    rt = df.groupby("n_agents")[["coop_runtime_ms"]].mean()
    rt_cbs = df[df.cbs_success].groupby("n_agents")["cbs_runtime_ms"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rt.index, rt["coop_runtime_ms"], marker="o", label="Cooperative A* (prioritized)", color="#2b6cb0")
    ax.plot(rt_cbs.index, rt_cbs.values, marker="o", label="CBS (optimal)", color="#805ad5")
    ax.set_yscale("log")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Mean planning time (ms, log scale)")
    ax.set_title("Planning time: prioritized vs. optimal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cbs_vs_coop_runtime.png"), dpi=150)
    plt.close(fig)

    summary = pd.DataFrame({
        "cbs_success_rate": cbs_rate,
        "mean_coop_soc_suboptimality": subopt,
        "mean_coop_makespan_suboptimality": makespan_subopt,
        "mean_cbs_ct_nodes_expanded": ct_nodes,
        "mean_coop_runtime_ms": rt["coop_runtime_ms"],
        "mean_cbs_runtime_ms": rt_cbs,
    }).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, "cbs_vs_cooperative_summary.csv"))
    print(summary)
    return summary


if __name__ == "__main__":
    df = run_all()
    make_plots(df)
    print("Done.")
