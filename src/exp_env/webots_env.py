"""
Interface between a running Webots simulation and the RL agents.

Each agent perceives a discrete state (e.g. gap-to-car-ahead) and
chooses an action (COOPERATE = yield / DEFECT = push forward).
Rewards mirror the MASD payoff matrix used in masd_env.py.

This setup follows the design for N=3, M=3 as described for Figure 4 in the paper, but can be easily extended to more agents or different action spaces.

In a M=3 setup, there are 4 distinct actions:
  - 0: Full Defection (0 units contributed) → Aggressive driving / No yielding
        The car prioritizes its "self-interested goal," which translates to pushing forward or maintaining maximum speed to maximize its own utility.
  - 1: Partial Cooperation (1 unit contributed) → Slightly more yielding
        The car allocates all its resources to the "group goal," which translates to yielding or slowing down significantly to ensure the overall traffic flow remains stable.
  - 2: Moderate Cooperation (2 units contributed) → More yielding
        The car allocates some resources to the "group goal," which translates to yielding or slowing down moderately to help maintain traffic flow while still considering its own progress.
  - 3: Full Cooperation (3 units contributed) → Maximum yielding / Safe driving
        The car allocates all its resources to the "group goal," which translates to yielding or slowing down significantly to ensure the overall traffic flow remains stable.
"""

from __future__ import annotations
from typing import Dict

# =========================
# STATES
# =========================
CLOSE  = 0   # gap < 4 m   → risky to push forward
MEDIUM = 1   # 4–8 m
FAR    = 2   # > 8 m       → safe to move

def _discretise_gap(gap_m: float) -> int:
    if gap_m < 4.0:
        return CLOSE
    elif gap_m < 8.0:
        return MEDIUM
    return FAR

class WebotsEnv:
    """
    Thin wrapper that translates Webots sensor readings into
    (state, reward) tuples consumable by any BaseAgent subclass.

    Usage inside car_controller.py:
        env = WebotsEnv(n_agents=3)
        state  = env.get_state(agent_id, gap_to_leader)
        reward = env.step(actions_dict)   # dict {agent_id: action}
    """

    def __init__(self, N: int = 3, M: int = 3, k: float = 2/3):
        self.N = N
        self.M = M        # Actions: 0, 1, ..., M units of cooperation
        self.k = k        # Selfishness factor
        self._last_actions: Dict[int, int] = {}

    def get_state(self, agent_id: int, obs) -> int:
        """
        Translates sensor data into states. 
        Returns a single integer state for the given agent.
        """
        front = min(obs[0], obs[1])
        return _discretise_gap(front)

    def step(self, actions: Dict[int, int]) -> Dict[int, float]:
        """
        Given a dict of {agent_id: action} for all agents,
        compute and return {agent_id: reward}.

        Payoff logic:
          Pi(a) = [ (1/N) * sum(aj) ] - [ (k * ai) / (M * (1-k)) ]
        """
        self._last_actions = dict(actions)
        n_cooperate = sum(actions.values())

        rewards: Dict[int, float] = {}
        for agent_id, ai in actions.items():
            # Pi(a) = [ (1/N) * sum(aj) ] - [ (k * ai) / (M * (1-k)) ]
            # Note: We divide total_contribution by (N*M) to normalize reward
            benefit = n_cooperate / (self.N * self.M)
            cost = (self.k * ai) / (self.M * (1.0 - self.k))
            
            rewards[agent_id] = benefit - cost
            
        return rewards

    @property
    def n_actions(self) -> int:
        return self.M + 1  # 0, 1, 2, 3 units

    def action_to_speed(self, action: int, max_speed: float = 13.0) -> float:
        """
        Maps contribution units to target speed.
        Action 3 (Full Cooperation) -> Slower/Safe speed to foster group flow.
        Action 0 (Full Defection)   -> Max speed / Aggressive driving.
        """
        # Linear mapping: more units contributed = more 'yielding' (lower speed)
        speed_reduction = (action / self.M) * 0.5
        return max_speed * (1.0 - speed_reduction)