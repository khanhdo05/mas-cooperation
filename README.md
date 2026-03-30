# Learning to Cooperate in Multi-Agent Social Dilemmas (Reproduction)

This project aims to reproduce the findings of ["Learning to Cooperate in Multi-Agent Social Dilemmas" (Munoz de Cote et al., 2006)](https://chercheurs.lille.inria.fr/lazaric/Webpage/Publications_files/munoz2006learning.pdf). The core objective is to demonstrate how two design principles—Change or Learn Fast (CoLF) and Change and Keep (CK)—can be integrated into standard Q-learning to help self-interested agents reach Pareto Efficient (PE) solutions in social dilemmas, rather than settling for sub-optimal Nash Equilibria (NE).

## Project Overview

Standard Q-learning often fails in multi-agent settings because the environment becomes non-stationary as all agents learn and change their policies simultaneously. This project implements:

- CoLF Principle: Uses a variable learning rate to account for non-stationarity caused by other agents.

- CK Principle: Uses a finite-state machine to repeat new actions, giving other agents time to react and providing more "informative" payoffs for Q-table updates.

## Repo Structure

```sh
/mas-cooperation
│
├── README.md               # You are here
├── requirements.txt        # numpy, matplotlib, etc.
├── main.py                 # ENTRY POINT: Orchestrates Experiment and Visualization
│
├── src/                    
│   ├── env/                
│   │   └── masd_env.py     # MASD payoff logic (N=3, M=3, k=2/3)
│   │   
│   ├── agents/             # RL Algorithms
|   │   ├── base_agent.py   # Abstract class for shared logic (Q-values, exploration)
│   │   ├── q_learning.py   # Algorithm 1: Standard Q-learning
│   │   ├── colf.py         # Algorithm 2: Change or Learn Fast
│   │   ├── ck.py           # Algorithm 3: Change and Keep
│   │   └── ck_colf.py      # Algorithm 4: Hybrid Logic
│   │
│   ├── experiment.py       # CLASS: Manages 100-trial batches & data logging 
│   └── data_viz.py         # CLASS: Generates plots (Moving Averages) like Fig 4
│
├── results/                
│   ├── data/               # Raw logs (CSV/JSON)
│   └── plots/              # Final reproduction graphics (PNG/PDF)
│
└── docs/                   # Poster and Paper materials
```

## Getting Started

### Clone and get into the repo

```sh
git clone git@github.com:khanhdo05/q-learning.git
cd mas-cooperation
```

### Create and use virtual environment for Python

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### Download deps

```sh
pip install -r requirements.txt
```

If you add any new dependencies, add them to the `requirements.txt` by this command:

```sh
pip freeze > requirements.txt
```

### Run main.py

```sh
python main.py
```

## Reproduction

### Target

We focus on reproducing **Figure 4**, which compares the performance and learning speed of the four algorithms in a medium-sized MASD game.

### Experimental Parameters

- **Agents (N)** (defined in `main.py`): 3
- **Actions (M)** (defined in `main.py`): 4 (Resource units {0,1,2,3})
- **Selfishness Factor (k)** (defined in `main.py`): 2/3
- **Discount Factor (γ)**: 0.95 (Except for CoLF should use 0.1)
- **Initial Q-values** (defined in `src/agents/base_agent.py`): Vmax = (r max)/(1-γ)
- **Exploration** (defined in `src/agents/base_agent.py`): epsilon-greedy, decaying from 0.2 to 0: max(0.2 - 0.00006t, 0)
- **Trials** (defined in `main.py`): Results are averaged over **100 independent trials**
- **Episodes** (defined in `main.py`): 200,000 per trial

## Extension

---future work---