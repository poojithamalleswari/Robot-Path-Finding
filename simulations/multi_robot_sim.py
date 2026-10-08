"""
multi_robot_sim.py

Animated demo for Question 2. Same grid, same robots, same start/goal
pairs in both panels. Left: every robot follows its own independently
planned A* path, and red starbursts mark each collision (two robots in
one cell, or two robots swapping through each other). Right: the
cooperative space-time A* plan, where robots wait or sidestep so the
collision counter stays at zero.

The script scans seeds until it finds an instance where independent
planning really does collide and cooperative planning succeeds, so the
demo is always a meaningful one.

Output: simulations/output/multi_robot_collisions.gif
"""

import os
import random
import sys

import numpy as np
import pygame

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "multi_robot"))
sys.path.insert(0, os.path.join(HERE, "..", "single_robot"))

import render_utils as ru
from cooperative_astar import plan_cooperative
from environment import make_multi_instance
from independent_astar import count_collisions, plan_independent

CELL = 24
HUD = 44
SIZE = 18
DENSITY = 0.12
N_AGENTS = 8
SUBSTEPS = 4


def pad(paths, length):
    return [p + [p[-1]] * (length - len(p)) for p in paths]


def collision_events(padded):
    """List of (time, (x_cell, y_cell)) with half-steps for swaps."""
    events = []
    T = len(padded[0])
    n = len(padded)
    for t in range(T):
        seen = {}
        for i in range(n):
            seen.setdefault(padded[i][t], []).append(i)
        for pos, agents in seen.items():
            if len(agents) > 1:
                events.append((float(t), pos))
    for t in range(T - 1):
        for i in range(n):
            for j in range(i + 1, n):
                if (padded[i][t] == padded[j][t + 1] and padded[i][t + 1] == padded[j][t]
                        and padded[i][t] != padded[i][t + 1]):
                    a, b = padded[i][t], padded[i][t + 1]
                    events.append((t + 0.5, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)))
    return events


def find_instance():
    for seed in range(1, 400):
        rng = random.Random(seed)
        grid, starts, goals = make_multi_instance(SIZE, DENSITY, N_AGENTS, rng)
        naive = plan_independent(grid, starts, goals)
        v, e = count_collisions(naive["paths"])
        if v + e < 3:
            continue
        coop = plan_cooperative(grid, starts, goals)
        if coop["success"]:
            return seed, grid, starts, goals, naive["paths"], coop["paths"]
    raise RuntimeError("no suitable demo instance found")


def main():
    seed, grid, starts, goals, naive_paths, coop_paths = find_instance()
    T = max(max(len(p) for p in naive_paths), max(len(p) for p in coop_paths))
    naive_pad, coop_pad = pad(naive_paths, T), pad(coop_paths, T)
    events = collision_events(naive_pad)

    f = ru.font(28)
    f_small = ru.font(22)
    panels = {k: ru.init(SIZE, SIZE, CELL, HUD) for k in ("naive", "coop")}

    def interp(path, tau):
        i = min(int(tau), len(path) - 1)
        j = min(i + 1, len(path) - 1)
        frac = tau - int(tau)
        a, b = ru.cell_center(path[i], CELL, HUD), ru.cell_center(path[j], CELL, HUD)
        return (a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac)

    def draw(surface, padded, title, tau, show_events):
        ru.draw_grid(surface, grid, CELL, HUD)
        for i in range(N_AGENTS):
            col = ru.ROBOT_COLORS[i % len(ru.ROBOT_COLORS)]
            ru.draw_start(surface, ru.cell_center(starts[i], CELL, HUD), CELL, col)
            ru.draw_star(surface, ru.cell_center(goals[i], CELL, HUD), CELL * 0.42, col)
        for i in range(N_AGENTS):
            col = ru.ROBOT_COLORS[i % len(ru.ROBOT_COLORS)]
            ru.draw_robot(surface, interp(padded[i], tau), CELL, col, label=i + 1, f=f_small)
        count = 0
        if show_events:
            for te, loc in events:
                if te <= tau:
                    count += 1
                age = tau - te
                if 0 <= age < 1.2:
                    cx = ru.cell_center(loc, CELL, HUD)
                    ru.draw_burst(surface, cx, CELL, 1 - age / 1.2)
        ru.draw_hud(surface, title, f, f"collisions: {count}")

    frames = []
    total = (T - 1) * SUBSTEPS + 1
    for k in range(total):
        tau = k / SUBSTEPS
        draw(panels["naive"], naive_pad, "Independent A*", tau, True)
        draw(panels["coop"], coop_pad, "Cooperative A*", tau, False)
        gap = np.full((panels["naive"].get_height(), 6, 3), ru.BG, dtype=np.uint8)
        frames.append(np.hstack([ru.capture(panels["naive"]), gap, ru.capture(panels["coop"])]))

    out = os.path.join(HERE, "output", "multi_robot_collisions.gif")
    ru.save_gif(frames, out, fps=12, hold_last=20)
    print("wrote", out, "seed", seed, "T", T, "collision events:", len(events),
          "naive cost", sum(len(p) - 1 for p in naive_paths), "coop cost", sum(len(p) - 1 for p in coop_paths))


if __name__ == "__main__":
    main()
