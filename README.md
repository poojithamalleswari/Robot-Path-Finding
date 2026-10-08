# Robot Path-Finding with A* Search

Two assignment questions, plus three extensions and three animated simulations.

| Part | What it does | Folder |
|---|---|---|
| **Q1** | A* on random grids, compared across four heuristics | `single_robot/` |
| **Q1+** | Weighted A* speed-vs-quality sweep; paired significance tests, box plots, node-expansion heatmaps | `single_robot/` |
| **Q2** | Several robots on one grid: independent A* (collides) vs. cooperative space-time A* (proposed fix) | `multi_robot/` |
| **Q2+** | Conflict-Based Search (CBS) as the optimal yardstick for the cooperative planner | `multi_robot/` |
| **Q3 (extension)** | Robot with a limited sensor that replans as it discovers obstacles | `dynamic_robot/` |
| **Simulations** | Pygame-rendered GIFs of all of the above | `simulations/` |

The full write-up (problem definition, assumptions, algorithms, setup, results,
limitations, references) is in [`report/Robot_Pathfinding_Report.docx`](report/Robot_Pathfinding_Report.docx)
(a PDF copy sits next to it). This README is about running the code.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # every experiment, analysis and simulation
python run_all.py --no-sim   # same, but skip the Pygame GIFs
```

Everything is seeded and deterministic, so reruns reproduce the same node counts, costs,
collision counts and figures. Only measured runtimes vary between runs and machines (and,
rarely, a CBS instance sitting right at its 3-second budget). Each script can also be run on its own.
`single_robot/analysis.py` must run after `single_robot/experiment.py` because it
reads that script's CSV; `run_all.py` handles the order.

Pygame runs headless (no window needed), so this works on a server or in CI.

## Layout

```
robot-pathfinding/
├── README.md
├── requirements.txt
├── run_all.py
├── single_robot/
│   ├── grid_generator.py        random solvable grids (BFS-checked)
│   ├── heuristics.py            manhattan / euclidean / chebyshev / weighted
│   ├── astar.py                 A* with expansion/generation counters
│   ├── experiment.py            1,080-run heuristic comparison + plots
│   ├── analysis.py              paired t / Wilcoxon tests, box plots, heatmaps
│   ├── tiebreak_ablation.py     effect of the A* tie-breaking rule
│   ├── weighted_astar_sweep.py  weights 1.0-3.0 trade-off curve
│   └── results/                 CSVs + PNGs
├── multi_robot/
│   ├── environment.py           random multi-robot instances
│   ├── independent_astar.py     naive per-robot A* + collision checker
│   ├── spacetime_astar.py       shared (cell, time) A* low-level search
│   ├── cooperative_astar.py     prioritized planning with a reservation table
│   ├── cbs.py                   Conflict-Based Search
│   ├── experiment.py            independent vs. cooperative, 2-10 robots
│   ├── cbs_experiment.py        cooperative vs. CBS, 2-12 robots
│   └── results/
├── dynamic_robot/
│   ├── repeated_replanning.py   limited-sensor navigator (replans with A*)
│   ├── experiment.py            sensor radius x obstacle density sweep
│   └── results/
├── simulations/
│   ├── render_utils.py          shared Pygame drawing helpers
│   ├── single_robot_sim.py      three heuristics searching side by side
│   ├── dynamic_robot_sim.py     fog-of-war + sensor + REPLANNING banner
│   ├── multi_robot_sim.py       collisions (left) vs. cooperative plan (right)
│   ├── export_stills.py         stills from the GIFs for the report
│   └── output/                  *.gif and stills/
└── report/
    ├── compute_numbers.py       derives every number the report quotes -> numbers.json
    ├── build_report.js          builds the .docx from numbers.json + the figures
    ├── Robot_Pathfinding_Report.docx
    └── Robot_Pathfinding_Report.pdf
```

## The simulations

Open these in any browser or image viewer:

- `simulations/output/single_robot_search.gif`: Manhattan, Euclidean and Chebyshev A*
  on the same grid. Cells light up in the order they are expanded, at the same
  speed in each panel, so the better heuristic visibly finishes first.
- `simulations/output/dynamic_replanning.gif`: the robot starts blind (dark fog),
  its sensor reveals nearby cells, and it replans whenever something blocks its route.
- `simulations/output/multi_robot_collisions.gif`: eight robots, independent A* on the
  left (red starbursts = collisions) and cooperative A* on the right.

## Headline results

(All numbers below are produced by the scripts in this repo; `report/numbers.json` holds the
exact values the report quotes.)

**Q1, heuristics (1,080 runs).** Manhattan expands 83.1 nodes on average,
against 242.5 for Euclidean and 268.0 for Chebyshev (34% and
31% as many), with identical (optimal) path cost. All pairwise differences are significant
(paired t-test and Wilcoxon; largest p-value 8.6e-13). 1.5x-weighted Manhattan expands the fewest
nodes (54.7) but its paths are about 2.2% longer.

**Tie-breaking matters.** The same Manhattan A* expands 83.1 nodes when ties between
equal-f nodes go to the node closest to the goal, but 174.2 (2.10x as many)
with first-in-first-out ties, at identical path cost. Euclidean and Chebyshev are barely affected.

**Weighted A\*.** Weight 1.1 cuts nodes expanded by 20% for a 0.15% cost increase; weight 1.5
gives 42% fewer nodes for 2.7% longer paths; beyond that the saving flattens while quality keeps degrading.

**Q2, multi-robot (200 instances).** Independent A* is collision-free 95% of the time with
2 robots but only 7.5% with 10. Cooperative A* was collision-free in all
200 instances, for 0.74% extra total path cost at 10 robots.

**CBS comparison (192 instances).** Cooperative A* is on average at most 0.48% above the optimal
sum-of-costs (exactly optimal on 160 of the 186 instances both methods solved, at most 6.1% worse
otherwise). CBS's constraint tree grows from about 1 node at 2 robots to about 93 at
12, and it solved 87.5% of 12-robot instances within its budget.

**Partial observability.** A robot that senses 1 cell ahead travels about 42% farther than one with the
full map; at radius 8 the penalty is about 4.5%.

## Reading the results

- `*_summary.csv` are the aggregate tables used in the report; `*_results.csv` are
  the raw one-row-per-run data if you want to slice them yourself, for example:

  ```python
  import pandas as pd
  df = pd.read_csv("single_robot/results/single_robot_results.csv")
  df.groupby(["density", "heuristic"])["nodes_expanded"].mean().unstack()
  ```
- Compare *nodes expanded* between heuristics (it is what the heuristic actually
  controls) and use *optimality ratio* to check solution quality; raw path cost is not
  comparable across different grids.
- `significance_tests.csv` reports effect size (mean difference) next to the p-values.
  With hundreds of paired samples the p-values get tiny easily, so judge importance by
  the size of the difference.

## Things to know (limitations)

- Maps are uniform random obstacles; no structured or benchmark maps.
- The CBS here uses one simplification (a robot that runs into a parked, finished robot
  is the only one branched on). See Section 3.5 of the report.
- The replanning robot uses repeated A*, not true D* Lite, so it matches D* Lite's
  behaviour but not its per-replan efficiency.
- The CBS sweep has 8 instances per configuration, so its percentages are coarse.
- The multi-robot planners use first-in-first-out tie-breaking internally; the tie-break study covers the single-robot A* only.

## Rebuilding the report

`run_all.py` already runs `report/compute_numbers.py`, which reads the result CSVs and writes
`report/numbers.json`. It also asserts the report's qualitative claims (for example that
Manhattan expands fewer nodes than Euclidean), so it stops with an error if a rerun ever
contradicts the prose. Then:

```bash
cd report
npm install docx
node build_report.js
```

## Pushing to GitHub

```bash
git init
git add .
git commit -m "A* heuristics, multi-robot planning (cooperative A*, CBS), replanning, simulations"
git branch -M main
git remote add origin https://github.com/<your-username>/robot-pathfinding-astar.git
git push -u origin main
```

Then put the repository URL into Section 8 of the report (it currently holds a placeholder)
and rebuild it.

## References

- Hart, Nilsson, Raphael (1968). A formal basis for the heuristic determination of minimum cost paths.
- Pohl (1970). Heuristic search viewed as path finding in a graph.
- Silver (2005). Cooperative pathfinding.
- Koenig, Likhachev (2002). D* Lite.
- Sharon, Stern, Felner, Sturtevant (2012). Conflict-based search for optimal multi-agent path finding.
- Stern et al. (2019). Multi-agent pathfinding: definitions, variants, and benchmarks.
