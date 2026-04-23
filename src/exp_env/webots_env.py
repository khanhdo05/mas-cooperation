"""
Interface between a running Webots simulation and the RL agents.

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

import numpy as np

from src.exp_env.masd_env import MASDEnv


class WebotsEnv(MASDEnv):
    """
    Webots-specific extension of MASDEnv.

    Inherits:
        reward logic
        state transitions
        joint action logic
        reset logic

    Adds:
        action_to_speed()
        sensor preprocessing
        Webots utilities
    """
    def __init__(
        self,
        N: int = 3,
        M: int = 3,
        k: float = 2/3,
        max_speed: float = 13.0
    ):

        super().__init__(N, M, k)

        self.max_speed = max_speed

    # =========================
    # WEBOTS ADDITIONS
    # =========================

    def action_to_speed(
        self,
        action: int,
        max_speed: float | None = None,
    ) -> float:
        """
        Convert discrete action → wheel speed.

        0 = stop
        M = max speed

        Maps linearly, so intermediate actions yield intermediate speeds. 
        """
        return max_speed * (
            1 - action / self.M
        )
    
    def read_gap(
        self,
        sensor,
        cap: float = 50.0,
    ) -> float:
        """
        Prevent extreme sensor outliers.
        """
        return min(
            sensor.getValue(),
            cap,
        )

