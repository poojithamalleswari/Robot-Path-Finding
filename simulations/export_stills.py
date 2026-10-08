"""
export_stills.py

Pulls a few representative still frames out of each simulation GIF so
they can be placed in the report (a Word document can't play a GIF).
Run after the three *_sim.py scripts.
"""
import os
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "stills")
os.makedirs(OUT, exist_ok=True)

# gif name -> {still name: fractional position through the animation}
PICKS = {
    "single_robot_search": {"search_midway": 0.25, "path_following": 0.88},
    "dynamic_replanning": {"fog_midway": 0.35, "fog_late": 0.80},
    "multi_robot_collisions": {"collision_midway": 0.5, "collision_late": 0.75},
}

for gif, picks in PICKS.items():
    frames = list(imageio.get_reader(os.path.join(HERE, "output", gif + ".gif")))
    for name, frac in picks.items():
        idx = min(int(len(frames) * frac), len(frames) - 1)
        imageio.imwrite(os.path.join(OUT, f"{gif}__{name}.png"), frames[idx])
        print("wrote", f"{gif}__{name}.png", frames[idx].shape)
