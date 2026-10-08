"""
run_all.py

Regenerates every result in the project, in dependency order:

    python run_all.py            # everything (a few minutes)
    python run_all.py --no-sim   # skip the Pygame GIFs

Each step is just the script you could also run on its own; this file
only saves typing. analysis.py has to come after experiment.py because
it reads that script's CSV.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Q1  heuristic comparison",        "single_robot/experiment.py"),
    ("Q1  significance / boxplots / heatmaps", "single_robot/analysis.py"),
    ("Q1  tie-breaking ablation", "single_robot/tiebreak_ablation.py"),
    ("Q1+ weighted A* trade-off sweep", "single_robot/weighted_astar_sweep.py"),
    ("Q2  independent vs cooperative A*", "multi_robot/experiment.py"),
    ("Q2+ CBS vs cooperative A*",       "multi_robot/cbs_experiment.py"),
    ("Q3  dynamic replanning sweep",    "dynamic_robot/experiment.py"),
]
SIM_STEPS = [
    ("sim single-robot search",   "simulations/single_robot_sim.py"),
    ("sim dynamic replanning",    "simulations/dynamic_robot_sim.py"),
    ("sim multi-robot collisions", "simulations/multi_robot_sim.py"),
    ("report stills from the GIFs", "simulations/export_stills.py"),
    ("report numbers (numbers.json)", "report/compute_numbers.py"),
]


def main():
    steps = STEPS if "--no-sim" in sys.argv else STEPS + SIM_STEPS
    for label, script in steps:
        print(f"\n=== {label}  ({script}) ===", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, script)], check=True,
                       env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
    print("\nAll done.")


if __name__ == "__main__":
    main()
