"""
experiment.py

Sweeps grid size, obstacle density, and robot count, comparing:
  - independent_astar  (plain A*, robots planned with no knowledge
                         of each other)
  - cooperative_astar  (prioritized, space-time A* with a reservation
                         table -- our proposed fix)

Results go to results/multi_robot_results.csv, with summary plots
alongside it.
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
from independent_astar import plan_independent, count_collisions
from cooperative_astar import plan_cooperative

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [20, 30]
DENSITIES = [0.10, 0.20]
AGENT_COUNTS = [2, 4, 6, 8, 10]
TRIALS_PER_CONFIG = 10
MASTER_SEED = 123


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

                    naive = plan_independent(grid, starts, goals)
                    v, e = count_collisions(naive["paths"])
                    coop = plan_cooperative(grid, starts, goals)

                    row = {
                        "grid_size": size,
                        "density": density,
                        "n_agents": n_agents,
                        "trial": trial,
                        "naive_vertex_collisions": v,
                        "naive_edge_collisions": e,
                        "naive_collision_free": (v == 0 and e == 0),
                        "naive_sum_of_costs": naive["sum_of_costs"],
                        "naive_runtime_ms": naive["runtime_ms"],
                        "coop_success": coop["success"],
                        "coop_sum_of_costs": coop["sum_of_costs"],
                        "coop_runtime_ms": coop["runtime_ms"],
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "multi_robot_results.csv"), index=False)
    return df


def make_plots(df):
    # 1. Collision-free rate: naive vs cooperative, as a function of agent count
    naive_rate = df.groupby("n_agents")["naive_collision_free"].mean()
    coop_rate = df.groupby("n_agents")["coop_success"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(naive_rate.index, naive_rate.values, marker="o", label="Independent A* (collision-free)", color="#c53030")
    ax.plot(coop_rate.index, coop_rate.values, marker="o", label="Cooperative A* (success)", color="#2b6cb0")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Solution quality vs. number of robots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "success_rate_vs_agents.png"), dpi=150)
    plt.close(fig)

    # 2. Mean number of collision events (naive) vs agent count
    coll = df.groupby("n_agents")[["naive_vertex_collisions", "naive_edge_collisions"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coll.index, coll["naive_vertex_collisions"], marker="o", label="Vertex collisions", color="#dd6b20")
    ax.plot(coll.index, coll["naive_edge_collisions"], marker="o", label="Edge (swap) collisions", color="#805ad5")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Mean collision events per trial (independent A*)")
    ax.set_title("Collisions grow with robot count under naive planning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "collisions_vs_agents.png"), dpi=150)
    plt.close(fig)

    # 3. Path-cost overhead of cooperative planning (only where both succeeded/collision-free-compared)
    valid = df[df.coop_success]
    overhead = (valid["coop_sum_of_costs"] - valid["naive_sum_of_costs"]) / valid["naive_sum_of_costs"]
    overhead_by_agents = overhead.groupby(valid["n_agents"]).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(overhead_by_agents.index.astype(str), overhead_by_agents.values, color="#2b6cb0")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Extra total path cost vs. naive plan (%)")
    ax.set_title("Cost of guaranteeing collision-free paths")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cost_overhead_vs_agents.png"), dpi=150)
    plt.close(fig)

    # 4. Runtime comparison
    rt = df.groupby("n_agents")[["naive_runtime_ms", "coop_runtime_ms"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rt.index, rt["naive_runtime_ms"], marker="o", label="Independent A*", color="#c53030")
    ax.plot(rt.index, rt["coop_runtime_ms"], marker="o", label="Cooperative A*", color="#2b6cb0")
    ax.set_xlabel("Number of robots")
    ax.set_ylabel("Mean planning time (ms)")
    ax.set_title("Planning time vs. number of robots")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "runtime_vs_agents.png"), dpi=150)
    plt.close(fig)

    summary = pd.DataFrame({
        "naive_collision_free_rate": naive_rate,
        "coop_success_rate": coop_rate,
        "mean_naive_vertex_collisions": coll["naive_vertex_collisions"],
        "mean_naive_edge_collisions": coll["naive_edge_collisions"],
        "mean_cost_overhead_pct": overhead_by_agents,
        "mean_naive_runtime_ms": rt["naive_runtime_ms"],
        "mean_coop_runtime_ms": rt["coop_runtime_ms"],
    }).round(3)
    summary.to_csv(os.path.join(RESULTS_DIR, "multi_robot_summary.csv"))
    print(summary)
    return summary


def save_example_visual(size=18, density=0.15, n_agents=5, seed=9):
    """Side-by-side snapshot of naive vs cooperative paths, for the report."""
    rng = random.Random(seed)
    grid, starts, goals = make_multi_instance(size, density, n_agents, rng)
    naive = plan_independent(grid, starts, goals)
    coop = plan_cooperative(grid, starts, goals)

    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, result, title in [
        (axes[0], naive["paths"], "Independent A* (may collide)"),
        (axes[1], coop["paths"] if coop["success"] else naive["paths"],
         "Cooperative A* (collision-free)" if coop["success"] else "Cooperative A* (failed)"),
    ]:
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        for i, path in enumerate(result):
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, color=colors[i % 10], linewidth=2)
            ax.scatter([xs[0]], [ys[0]], color=colors[i % 10], marker="o", s=90, zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], color=colors[i % 10], marker="*", s=160, zorder=5)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Multi-robot example: {n_agents} robots on a {size}x{size} grid (density={density})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "example_multi_robot.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = run_all()
    make_plots(df)
    save_example_visual()
    print("Done. Results in", RESULTS_DIR)
