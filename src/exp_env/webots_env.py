"""
webots_env.py
Interface between a running Webots simulation and the RL agents.

Each agent perceives a discrete state (e.g. gap-to-car-ahead) and
chooses an action (COOPERATE = yield / DEFECT = push forward).
Rewards mirror the MASD payoff matrix used in masd_env.py.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple


# ── Payoff constants (keep in sync with masd_env.py) ────────────────────────
# N=3 agents, cooperation threshold k=2 out of 3
# If >= k agents cooperate → all cooperators get R, defectors get T
# else → cooperators get S, defectors get P
R = 3.0   # Reward  (mutual cooperation)
P = 1.0   # Punishment (mutual defection)
T = 5.0   # Temptation (defect while others cooperate)
S = 0.0   # Sucker  (cooperate while others defect)
K = 2     # cooperation threshold (out of N=3)

COOPERATE = 0
DEFECT = 1


# ── State discretisation ────────────────────────────────────────────────────
CLOSE  = 0   # gap < 3 m   → risky to push forward
MEDIUM = 1   # 3–8 m
FAR    = 2   # > 8 m       → safe to move

def _discretise_gap(gap_m: float) -> int:
    if gap_m < 3.0:
        return CLOSE
    elif gap_m < 8.0:
        return MEDIUM
    return FAR


# ── WebotsEnv ────────────────────────────────────────────────────────────────
class WebotsEnv:
    """
    Thin wrapper that translates Webots sensor readings into
    (state, reward) tuples consumable by any BaseAgent subclass.

    Usage inside car_controller.py:
        env = WebotsEnv(n_agents=3)
        state  = env.get_state(agent_id, gap_to_leader)
        reward = env.step(actions_dict)   # dict {agent_id: action}
    """

    def __init__(self, n_agents: int = 3):
        self.n_agents = n_agents
        self._last_actions: Dict[int, int] = {}

    # ── observation ──────────────────────────────────────────────────────────
    def get_state(self, agent_id: int, gap_to_leader_m: float) -> int:
        """
        Returns a single integer state for the given agent.
        Currently: discretised gap to the car directly ahead.
        """
        return _discretise_gap(gap_to_leader_m)

    # ── transition & reward ──────────────────────────────────────────────────
    def step(self, actions: Dict[int, int]) -> Dict[int, float]:
        """
        Given a dict of {agent_id: action} for all agents,
        compute and return {agent_id: reward}.

        Payoff logic (MASD, N=3, k=2):
          cooperators_count >= K  → cooperators get R, defectors get T
          cooperators_count  < K  → cooperators get S, defectors get P
        """
        self._last_actions = dict(actions)
        n_cooperate = sum(1 for a in actions.values() if a == COOPERATE)

        rewards: Dict[int, float] = {}
        for agent_id, action in actions.items():
            if n_cooperate >= K:
                rewards[agent_id] = R if action == COOPERATE else T
            else:
                rewards[agent_id] = S if action == COOPERATE else P

        return rewards

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def n_states(self) -> int:
        return 3   # CLOSE / MEDIUM / FAR

    @property
    def n_actions(self) -> int:
        return 2   # COOPERATE / DEFECT

    def action_to_speed(self, action: int, base_speed: float = 10.0) -> float:
        """
        Maps a discrete action to a wheel speed (m/s).
          COOPERATE → slow down / yield
          DEFECT    → maintain / accelerate
        """
        if action == COOPERATE:
            return base_speed * 0.5     # yield: half speed
        return base_speed               # defect: full speed