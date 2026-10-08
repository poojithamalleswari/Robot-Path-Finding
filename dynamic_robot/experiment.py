"""
experiment.py (dynamic_robot)

Sweeps obstacle density and sensor radius, and measures the "cost of
partial observability": how much farther the robot has to travel when
it can only see `sensor_radius` cells around itself, compared to a
robot that's handed the whole map up front (the single_robot/ A*
baseline). We'd expect that cost to shrink as sensor_radius grows,
since a robot that can see further ahead has less chance of walking
into a surprise and needing to backtrack.
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "single_robot"))
from astar import astar
from grid_generator import make_instance
from heuristics import manhattan
from repeated_replanning import navigate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZE = 35
DENSITIES = [0.10, 0.20, 0.30]
SENSOR_RADII = [1, 2, 3, 5, 8]
TRIALS_PER_CONFIG = 25
MASTER_SEED = 11


def run_all():
    rng = random.Random(MASTER_SEED)
    rows = []

    for density in DENSITIES:
        for trial in range(TRIALS_PER_CONFIG):
            grid, start, goal = make_instance(GRID_SIZE, density, rng)
            full_knowledge = astar(grid, start, goal, manhattan)
            optimal_cost = full_knowledge["cost"]

            for radius in SENSOR_RADII:
                result = navigate(grid, start, goal, sensor_radius=radius)
                rows.append({
                    "density": density,
                    "trial": trial,
                    "sensor_radius": radius,
                    "success": result["success"],
                    "distance_traveled": result["distance_traveled"],
                    "replans": result["replans"],
                    "optimal_cost": optimal_cost,
                    "overhead_ratio": (result["distance_traveled"] / optimal_cost
                                       if result["success"] and optimal_cost else None),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "dynamic_results.csv"), index=False)
    return df


def make_plots(df):
    ok = df[df.success]

    # 1. Distance overhead vs sensor radius
    overhead = ok.groupby("sensor_radius")["overhead_ratio"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(overhead.index, overhead.values, marker="o", color="#2b6cb0")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", label="Full-map optimal")
    ax.set_xlabel("Sensor radius (cells)")
    ax.set_ylabel("Distance traveled / full-map-optimal distance")
    ax.set_title("Cost of partial observability vs. sensor range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "overhead_vs_radius.png"), dpi=150)
    plt.close(fig)

    # 2. Replans vs density, by sensor radius
    fig, ax = plt.subplots(figsize=(7, 5))
    for radius in SENSOR_RADII:
        sub = ok[ok.sensor_radius == radius].groupby("density")["replans"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"radius={radius}")
    ax.set_xlabel("Obstacle density")
    ax.set_ylabel("Mean number of replans")
    ax.set_title("Replanning frequency vs. density and sensor range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "replans_vs_density.png"), dpi=150)
    plt.close(fig)

    # 3. Success rate vs sensor radius (can the robot even reach the goal
    #    with such limited vision that it talks itself into a dead end?)
    rate = df.groupby("sensor_radius")["success"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rate.index, rate.values, marker="o", color="#c53030")
    ax.set_xlabel("Sensor radius (cells)")
    ax.set_ylabel("Fraction of trials reaching the goal")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Navigation success rate vs. sensor range")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "success_vs_radius.png"), dpi=150)
    plt.close(fig)

    summary = pd.DataFrame({
        "mean_overhead_ratio": overhead,
        "success_rate": rate,
    }).round(4)
    summary.to_csv(os.path.join(RESULTS_DIR, "dynamic_summary.csv"))
    print(summary)
    return summary


if __name__ == "__main__":
    df = run_all()
    make_plots(df)
    print("Done.")
