"""
render_utils.py

Small Pygame helper layer shared by the three simulation scripts.

Pygame is normally used for live windows, but here it's run headless
(SDL's "dummy" video driver) so frames can be drawn off-screen, copied
into numpy arrays, and stitched into animated GIFs. That means the
simulations can be regenerated on any machine, with no display and no
screen recorder needed -- and the GIF can be dropped straight into a
report or slide deck.
"""

import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio
import numpy as np
import pygame

# ---- palette ----
BG = (18, 22, 33)
FREE = (236, 240, 245)
UNKNOWN = (58, 66, 84)
OBSTACLE = (28, 32, 44)
OBSTACLE_EDGE = (70, 78, 98)
GRID_LINE = (205, 212, 224)
TEXT = (230, 235, 245)
GOAL_COLOR = (255, 196, 61)
START_COLOR = (80, 200, 140)
FLASH_RED = (235, 64, 52)

ROBOT_COLORS = [
    (66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75),
    (145, 30, 180), (70, 200, 200), (240, 200, 30), (250, 120, 190),
    (128, 128, 0), (0, 128, 128), (170, 110, 40), (120, 120, 255),
]


def init(cols, rows, cell, hud=44):
    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((cols * cell, rows * cell + hud))
    return surface


def font(size):
    # Font(None, ...) uses pygame's bundled default font, so no system
    # fonts are needed. init() is idempotent, so it's safe to call here
    # in case a script builds a font before creating its first surface.
    pygame.font.init()
    return pygame.font.Font(None, size)


def draw_grid(surface, grid, cell, hud=44, known_mask=None, known_grid=None):
    """
    Draws the world. With known_mask/known_grid supplied (the dynamic
    robot's belief map), unseen cells are drawn dark "fog", and only
    obstacles the robot has actually sensed are drawn as walls.
    """
    rows, cols = grid.shape
    surface.fill(BG)
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(c * cell, hud + r * cell, cell, cell)
            if known_mask is not None and not known_mask[r, c]:
                pygame.draw.rect(surface, UNKNOWN, rect)
            elif (known_grid if known_grid is not None else grid)[r, c] == 1:
                pygame.draw.rect(surface, OBSTACLE, rect)
                pygame.draw.rect(surface, OBSTACLE_EDGE, rect, 1)
            else:
                pygame.draw.rect(surface, FREE, rect)
                pygame.draw.rect(surface, GRID_LINE, rect, 1)


def cell_center(cell_rc, cell, hud=44):
    r, c = cell_rc
    return (c * cell + cell / 2, hud + r * cell + cell / 2)


def draw_robot(surface, center, cell, color, label=None, f=None):
    x, y = int(center[0]), int(center[1])
    radius = int(cell * 0.38)
    pygame.draw.circle(surface, (0, 0, 0), (x + 2, y + 3), radius)          # shadow
    pygame.draw.circle(surface, color, (x, y), radius)
    pygame.draw.circle(surface, (255, 255, 255), (x - radius // 3, y - radius // 3), max(2, radius // 4))  # highlight
    pygame.draw.circle(surface, (20, 20, 20), (x, y), radius, 2)
    if label is not None and f is not None:
        img = f.render(str(label), True, (255, 255, 255))
        surface.blit(img, img.get_rect(center=(x, y)))


def draw_star(surface, center, size, color):
    pts = []
    for i in range(10):
        ang = math.pi / 2 + i * math.pi / 5
        rad = size if i % 2 == 0 else size * 0.45
        pts.append((center[0] + rad * math.cos(ang), center[1] - rad * math.sin(ang)))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (90, 60, 0), pts, 1)


def draw_start(surface, center, cell, color=START_COLOR):
    r = int(cell * 0.2)
    pygame.draw.circle(surface, color, (int(center[0]), int(center[1])), r, 3)


def draw_trail(surface, centers, color, width=3):
    if len(centers) >= 2:
        pygame.draw.lines(surface, color, False, centers, width)


def draw_burst(surface, center, cell, strength):
    """Collision marker: an expanding red starburst. strength in 0..1."""
    x, y = center
    outer = cell * (0.5 + 0.9 * strength)
    inner = outer * 0.5
    pts = []
    for i in range(16):
        ang = i * math.pi / 8
        rad = outer if i % 2 == 0 else inner
        pts.append((x + rad * math.cos(ang), y + rad * math.sin(ang)))
    pygame.draw.polygon(surface, FLASH_RED, pts)
    pygame.draw.polygon(surface, (255, 220, 120), pts, 2)


def draw_sensor_ring(surface, center, radius_cells, cell):
    # diamond (Manhattan ball) matching how the sensing is computed
    x, y = center
    d = (radius_cells + 0.5) * cell
    pts = [(x, y - d), (x + d, y), (x, y + d), (x - d, y)]
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(overlay, (80, 170, 255, 45), pts)
    pygame.draw.polygon(overlay, (80, 170, 255, 190), pts, 2)
    surface.blit(overlay, (0, 0))


def draw_hud(surface, text, f, right_text=None):
    img = f.render(text, True, TEXT)
    surface.blit(img, (10, 12))
    if right_text:
        img2 = f.render(right_text, True, (255, 196, 61))
        surface.blit(img2, (surface.get_width() - img2.get_width() - 10, 12))


def capture(surface):
    arr = pygame.surfarray.array3d(surface)      # (W, H, 3)
    return np.transpose(arr, (1, 0, 2)).copy()   # -> (H, W, 3)


def save_gif(frames, path, fps=12, hold_last=18):
    frames = list(frames) + [frames[-1]] * hold_last   # pause on the final frame
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
