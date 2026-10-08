"""
dynamic_robot_sim.py

Animated demo for the dynamic-replanning extension. The robot starts
knowing nothing: unseen cells are drawn as dark fog. As it drives, its
sensor (the translucent blue diamond) reveals nearby cells. The dashed
plan shows what it currently intends to do; whenever a newly-seen
obstacle blocks that plan, the screen flashes "REPLANNING" and a new
route is drawn.

Output: simulations/output/dynamic_replanning.gif
"""

import os
import random
import sys

import numpy as np
import pygame

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "single_robot"))
sys.path.insert(0, os.path.join(HERE, "..", "dynamic_robot"))

import render_utils as ru
from astar import astar
from grid_generator import make_instance
from heuristics import manhattan
from repeated_replanning import _sensed_cells

CELL = 22
HUD = 44
SIZE = 24
DENSITY = 0.26
RADIUS = 3
SEED = 4
SUBSTEPS = 4


def main():
    grid, start, goal = make_instance(SIZE, DENSITY, random.Random(SEED))
    known_grid = np.zeros_like(grid)
    known_mask = np.zeros_like(grid, dtype=bool)

    def sense(pos):
        for c in _sensed_cells(pos, RADIUS, SIZE):
            known_mask[c] = True
            known_grid[c] = grid[c]

    surface = ru.init(SIZE, SIZE, CELL, HUD)
    f = ru.font(28)
    f_big = ru.font(46)
    frames = []

    def render(px, trail, plan, replans, flash=False, step=0):
        ru.draw_grid(surface, grid, CELL, HUD, known_mask=known_mask, known_grid=known_grid)
        ru.draw_sensor_ring(surface, px, RADIUS, CELL)
        ru.draw_trail(surface, trail, (40, 100, 230), 4)
        if plan:
            pts = [px] + [ru.cell_center(p, CELL, HUD) for p in plan]
            # dashed look: draw short segments
            for a, b in zip(pts[:-1], pts[1:]):
                pygame.draw.line(surface, (255, 120, 40), a, b, 2)
        ru.draw_start(surface, ru.cell_center(start, CELL, HUD), CELL)
        ru.draw_star(surface, ru.cell_center(goal, CELL, HUD), CELL * 0.5, ru.GOAL_COLOR)
        ru.draw_robot(surface, px, CELL, (66, 135, 245))
        ru.draw_hud(surface, f"Sensor radius {RADIUS}   step {step}", f, f"replans: {replans}")
        if flash:
            txt = f_big.render("REPLANNING", True, ru.FLASH_RED)
            bg = pygame.Surface((txt.get_width() + 24, txt.get_height() + 12))
            bg.fill((0, 0, 0))
            bg.set_alpha(190)
            cx = surface.get_width() // 2
            surface.blit(bg, bg.get_rect(center=(cx, HUD + 40)))
            surface.blit(txt, txt.get_rect(center=(cx, HUD + 40)))
        frames.append(ru.capture(surface))

    pos = start
    sense(pos)
    plan_res = astar(known_grid, pos, goal, manhattan)
    plan = plan_res["path"][1:]
    replans = 1
    trail = [ru.cell_center(pos, CELL, HUD)]
    render(trail[0], trail, plan, replans, step=0)
    for _ in range(5):
        render(trail[0], trail, plan, replans, step=0)

    step = 0
    while pos != goal and step < 600:
        nxt = plan[0]
        a, b = ru.cell_center(pos, CELL, HUD), ru.cell_center(nxt, CELL, HUD)
        pos = nxt
        plan = plan[1:]
        step += 1
        sense(pos)
        invalid = any(known_mask[c] and known_grid[c] == 1 for c in plan)

        for s in range(SUBSTEPS):
            t = (s + 1) / SUBSTEPS
            px = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            render(px, trail + [px], plan, replans, step=step)
        trail.append(b)

        if invalid:
            for _ in range(5):
                render(b, trail, plan, replans, flash=True, step=step)
            res = astar(known_grid, pos, goal, manhattan)
            replans += 1
            if not res["found"]:
                break
            plan = res["path"][1:]
            for _ in range(4):
                render(b, trail, plan, replans, step=step)

    for _ in range(4):
        render(ru.cell_center(pos, CELL, HUD), trail, [], replans, step=step)

    out = os.path.join(HERE, "output", "dynamic_replanning.gif")
    ru.save_gif(frames, out, fps=14)
    print("wrote", out, "frames:", len(frames), "replans:", replans, "steps:", step, "reached:", pos == goal)


if __name__ == "__main__":
    main()
