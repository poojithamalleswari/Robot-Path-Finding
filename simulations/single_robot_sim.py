"""
single_robot_sim.py

Animated demo for Question 1. Three panels, same random grid, one
heuristic each (Manhattan / Euclidean / Chebyshev). Phase 1 shows the
search itself: cells light up in the order A* expands them, at the
same speed in every panel, so the heuristic that needs fewer
expansions visibly finishes first. Phase 2 sends a robot down the
(identical-length) path it found.

Output: simulations/output/single_robot_search.gif
"""

import heapq
import os
import random
import sys

import numpy as np
import pygame

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "single_robot"))

import render_utils as ru
from grid_generator import make_instance
from heuristics import HEURISTICS

CELL = 15
HUD = 44
SEED = 21
SIZE = 26
DENSITY = 0.22
PANELS = ["manhattan", "euclidean", "chebyshev"]
EXPAND_PER_FRAME = 6
SUBSTEPS = 3


def astar_trace(grid, start, goal, h):
    """A* (tie-break: lower h first, as in single_robot/astar.py) that records
    the order cells are expanded, plus the final path."""
    size = grid.shape[0]
    heap = [(h(start, goal), h(start, goal), 0, start)]
    g = {start: 0}
    parent = {start: None}
    order, counter = [], 1
    while heap:
        f, _, _, pos = heapq.heappop(heap)
        if f - h(pos, goal) > g[pos] + 1e-9:
            continue
        order.append(pos)
        if pos == goal:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (pos[0] + dr, pos[1] + dc)
            if 0 <= n[0] < size and 0 <= n[1] < size and grid[n] == 0:
                ng = g[pos] + 1
                if ng < g.get(n, 1e9):
                    g[n] = ng
                    parent[n] = pos
                    counter += 1
                    heapq.heappush(heap, (ng + h(n, goal), h(n, goal), counter, n))
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return order, path[::-1]


def draw_panel(surface, grid, start, goal, name, expanded, f, small, total, robot_px=None, trail=None, done=False):
    ru.draw_grid(surface, grid, CELL, HUD)
    for (r, c) in expanded:
        rect = pygame.Rect(c * CELL, HUD + r * CELL, CELL, CELL)
        pygame.draw.rect(surface, (255, 176, 92), rect)
        pygame.draw.rect(surface, (214, 140, 60), rect, 1)
    if trail:
        ru.draw_trail(surface, trail, (40, 100, 230), 4)
    ru.draw_start(surface, ru.cell_center(start, CELL, HUD), CELL)
    ru.draw_star(surface, ru.cell_center(goal, CELL, HUD), CELL * 0.48, ru.GOAL_COLOR)
    if robot_px is not None:
        ru.draw_robot(surface, robot_px, CELL, (66, 135, 245))
    label = name.capitalize()
    suffix = "  (goal reached)" if done else ""
    ru.draw_hud(surface, label, f, f"expanded: {len(expanded)}{suffix}")


def main():
    grid, start, goal = make_instance(SIZE, DENSITY, random.Random(SEED))
    traces = {n: astar_trace(grid, start, goal, HEURISTICS[n]) for n in PANELS}
    max_expand = max(len(t[0]) for t in traces.values())

    surfaces = {n: ru.init(SIZE, SIZE, CELL, HUD) for n in PANELS}
    f = ru.font(26)
    frames = []

    def compose(**kw):
        imgs = []
        for n in PANELS:
            order, path = traces[n]
            draw_panel(surfaces[n], grid, start, goal, n, kw["expanded"][n], f, None, len(order),
                       robot_px=kw.get("robot_px"), trail=kw.get("trail"), done=kw.get("done", False))
            imgs.append(ru.capture(surfaces[n]))
        gap = np.full((imgs[0].shape[0], 6, 3), ru.BG, dtype=np.uint8)
        row = imgs[0]
        for im in imgs[1:]:
            row = np.hstack([row, gap, im])
        return row

    # phase 1: search expansion
    n_frames = max_expand // EXPAND_PER_FRAME + 2
    for k in range(n_frames):
        count = (k + 1) * EXPAND_PER_FRAME
        expanded = {n: traces[n][0][:count] for n in PANELS}
        frames.append(compose(expanded=expanded))

    full = {n: traces[n][0] for n in PANELS}
    for _ in range(6):
        frames.append(compose(expanded=full))

    # phase 2: robots drive the path (all three paths have equal cost)
    path = traces[PANELS[0]][1]
    centers = [ru.cell_center(p, CELL, HUD) for p in path]
    for i in range(len(centers) - 1):
        for s in range(SUBSTEPS):
            t = (s + 1) / SUBSTEPS
            x = centers[i][0] + (centers[i + 1][0] - centers[i][0]) * t
            y = centers[i][1] + (centers[i + 1][1] - centers[i][1]) * t
            trail = centers[: i + 1] + [(x, y)]
            frames.append(compose(expanded=full, robot_px=(x, y), trail=trail,
                                  done=(i == len(centers) - 2 and s == SUBSTEPS - 1)))

    out = os.path.join(HERE, "output", "single_robot_search.gif")
    ru.save_gif(frames, out, fps=14)
    print("wrote", out, "frames:", len(frames), {n: len(traces[n][0]) for n in PANELS})


if __name__ == "__main__":
    main()
