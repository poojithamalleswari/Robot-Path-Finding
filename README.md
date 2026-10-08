# Robot Path-Finding with A* Search

A comparative study of A* pathfinding for single-robot and multi-robot environments. The project explores how different heuristics affect A* performance, how the problem changes when multiple robots have to share the same environment, and how path planning can be extended to environments where the robot has only limited knowledge of its surroundings.

---

# 1. Project Overview

The project starts with the standard single-robot pathfinding problem on a 2D grid and gradually extends it into more challenging scenarios.

The main stages are:

1. **Single-robot pathfinding using A\***
2. **Comparison of different heuristics**
3. **Tie-breaking strategies**
4. **Weighted A\***
5. **Multi-robot pathfinding**
6. **Cooperative A\* with space-time reservations**
7. **Conflict-Based Search (CBS)**
8. **Dynamic replanning with limited visibility**

The goal is not only to find a path, but to study the trade-off between:

- Path quality
- Number of nodes expanded
- Number of nodes generated
- Runtime
- Collision-free planning
- Computational overhead

---

# 2. Problem Definition

The environment is represented as a 2D grid containing:

- Free cells
- Obstacles
- Start position(s)
- Goal position(s)

A robot can move between neighbouring cells while avoiding obstacles.

For a single robot, the objective is to find a path from the start cell to the goal cell with minimum cost.

For multiple robots, each robot has its own start and goal. The robots must reach their respective goals without colliding with each other.

The project uses **4-directional movement**:

- Up
- Down
- Left
- Right

Each movement has a cost of 1.

---

# 3. Single-Robot A\*

The first part of the project implements the standard A* search algorithm.

A* evaluates a node using:

\[
f(n) = g(n) + h(n)
\]

where:

- \(g(n)\) = cost from the start node to the current node
- \(h(n)\) = estimated cost from the current node to the goal
- \(f(n)\) = estimated total cost of the path

The node with the lowest \(f(n)\) value is selected for expansion.

The implementation also keeps track of:

- Nodes expanded
- Nodes generated
- Runtime
- Final path cost
- Whether the solution is optimal

---

# 4. Heuristic Comparison

Three standard heuristics are compared for the single-robot problem.

## Manhattan Distance

For two points \((x_1,y_1)\) and \((x_2,y_2)\):

\[
h(n)=|x_1-x_2|+|y_1-y_2|
\]

Since the robot moves only horizontally and vertically, Manhattan distance is a natural heuristic for this problem.

## Euclidean Distance

\[
h(n)=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
\]

This represents the straight-line distance between the current node and the goal.

## Chebyshev Distance

\[
h(n)=\max(|x_1-x_2|,|y_1-y_2|)
\]

Although commonly useful when diagonal movement is allowed, it is also included here as a comparison.

### Metrics

Each heuristic is evaluated over multiple randomly generated grids using:

- Success rate
- Nodes expanded
- Nodes generated
- Runtime
- Path cost
- Optimality ratio

The results show that the choice of heuristic has a significant effect on the amount of search required.

---

# 5. Tie-Breaking

A* can have many nodes with the same \(f(n)\) value.

To study the effect of this, two tie-breaking strategies are compared:

### FIFO

Nodes with equal priority are processed in the order in which they were inserted.

### Lower \(h(n)\)

When two nodes have the same \(f(n)\), the node with the smaller heuristic value is preferred.

The second strategy generally reduces unnecessary exploration while producing the same path cost.

---

# 6. Weighted A\*

Weighted A* modifies the evaluation function to:

\[
f(n)=g(n)+w\cdot h(n)
\]

where \(w\) is the heuristic weight.

For:

\[
w=1
\]

Weighted A* becomes normal A*.

Increasing \(w\) gives the heuristic more influence, which generally causes the algorithm to explore fewer nodes and run faster, at the cost of potentially producing a slightly more expensive path.

The project evaluates several weights:

```text
1.0
1.1
1.2
1.5
1.75
2.0
2.5
3.0
```

This provides a direct view of the trade-off between search effort and solution quality.

---

# 7. Multi-Robot Pathfinding

The problem becomes significantly harder when multiple robots are introduced.

Each robot has:

```text
Start → Goal
```

and all robots operate on the same grid.

Simply running A* independently for every robot does not guarantee a valid solution because two individually valid paths can still conflict.

The main types of conflicts considered are:

### Vertex Collision

Two robots occupy the same cell at the same time.

### Edge Collision

Two robots exchange positions in the same timestep.

For example:

```text
t       t+1

A → B   B → A
```

Both individual paths are valid, but the robots collide while crossing.

---

# 8. Independent A\* vs Cooperative A\*

The first multi-robot approach is to run A* independently for every robot.

This is simple and fast, but it does not consider the paths of other robots.

To solve this problem, the project implements **Cooperative A\***.

Instead of searching only in the spatial grid, the search considers both:

\[
(x,y,t)
\]

where \(t\) represents the timestep.

A **reservation table** stores cells that are already occupied by previously planned robots.

When planning a new robot, the algorithm avoids states that would result in a collision.

This produces collision-free paths while introducing additional computational overhead.

---

# 9. Conflict-Based Search (CBS)

Cooperative A* depends on the order in which robots are planned.

To explore a different multi-agent approach, the project also implements **Conflict-Based Search (CBS)**.

CBS works in two levels.

### High Level

The algorithm searches for conflicts between robot paths.

When a conflict is found, constraints are added to prevent that conflict.

### Low Level

A* is then used to find a new path for the affected robot while respecting the new constraint.

The process continues until a set of collision-free paths is obtained.

The project compares CBS against Cooperative A* using:

- Success rate
- Sum of path costs
- Makespan
- Constraint-tree nodes
- Runtime

This also demonstrates an important trade-off: CBS can provide stronger conflict resolution, but its computational cost can increase as the number of robots grows.

---

# 10. Dynamic Replanning

The final extension considers a robot that does not know the complete environment beforehand.

Instead, the robot has a limited sensing radius.

It initially plans a path using the information available to it. As it moves, it discovers previously unknown obstacles and replans when necessary.

The process is approximately:

```text
Sense environment
       ↓
Plan path using A*
       ↓
Move along path
       ↓
Discover new information
       ↓
Replan if necessary
       ↓
Continue until goal
```

Different sensor radii are tested to study how additional visibility affects the amount of replanning required.

---

# 11. Experiments

The project contains separate experiments for the different stages of the problem.

```text
experiments/
├── q1/
│   ├── heuristic comparison
│   ├── statistical analysis
│   ├── tie-breaking
│   └── weighted A*
│
├── q2/
│   ├── independent A*
│   ├── cooperative A*
│   └── CBS
│
└── q3/
    └── dynamic replanning
```

The experiments generate numerical results that are used to compare the different approaches.

Random environments are generated using fixed seeds where reproducibility is required.

---

# 12. Visualizations

The project also contains visual simulations of the algorithms.

Generated visualizations include:

```text
simulations/output/
├── single_robot_search.gif
├── multi_robot_collisions.gif
├── dynamic_replanning.gif
└── ...
```

These provide a visual representation of:

- A* search
- Robot movement
- Multi-robot conflicts
- Collision avoidance
- Dynamic obstacle discovery
- Replanning

---

# 13. Project Structure

```text
robot-pathfinding/
│
├── algorithms/
│   ├── astar.py
│   ├── weighted_astar.py
│   ├── cooperative_astar.py
│   └── cbs.py
│
├── experiments/
│   ├── q1/
│   ├── q2/
│   └── q3/
│
├── simulations/
│   └── output/
│
├── report/
│
├── run_all.py
├── requirements.txt
└── README.md
```

---

# 14. Running the Project

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run all experiments:

```bash
python run_all.py
```

Individual experiment and simulation scripts can also be executed separately.

---

# 15. Results and Observations

The experiments show a few clear patterns.

### Single Robot

The heuristic has a major effect on search effort. Manhattan distance performs particularly well for the 4-directional grid because it closely matches the actual movement model.

### Weighted A\*

Increasing the heuristic weight generally reduces the number of nodes explored and runtime, but the resulting path can become slightly more expensive.

### Multiple Robots

Independent A* is fast but does not guarantee collision-free paths. As the number of robots increases, the probability of conflicts increases significantly.

Cooperative A* resolves these conflicts using time-aware planning and reservation tables.

### CBS

CBS provides another way of resolving conflicts by explicitly reasoning about collisions and constraints. However, the cost of conflict resolution can grow as the number of robots increases.

### Dynamic Replanning

A larger sensing radius reduces the amount of replanning required because the robot has more information about its surroundings before committing to a path.

---

# 16. The Overall Idea

The project follows a simple progression:

```text
Single Robot
     ↓
How well does A* work?
     ↓
Different heuristics
     ↓
Can we make A* faster?
     ↓
Weighted A* + tie-breaking
     ↓
Multiple Robots
     ↓
Independent paths cause collisions
     ↓
Cooperative A* / CBS
     ↓
Unknown / Dynamic Environment
     ↓
Limited sensing + replanning
```

The main idea is to start with basic A* pathfinding and gradually introduce the problems that appear when the environment becomes more realistic.

Rather than treating pathfinding as just finding a shortest path, the project looks at the different trade-offs involved in **search efficiency, solution quality, collision avoidance, scalability, and replanning**.

---

# 17. References

- Hart, P. E., Nilsson, N. J., & Raphael, B. — A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
- Sharon, G., Stern, R., Felner, A., & Sturtevant, N. — Conflict-Based Search for Optimal Multi-Agent Pathfinding.
- Silver, D. — Cooperative Pathfinding.
- Russell, S. & Norvig, P. — Artificial Intelligence: A Modern Approach.