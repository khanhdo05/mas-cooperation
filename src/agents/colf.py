from .base_agent import BaseAgent
import numpy as np


class CoLFAgent(BaseAgent):
    """
    Change or Learn Fast
    """
    def __init__(
        self,
        agent_id,
        state_size,
        action_size,
        gamma,
        seed,
        alpha_ns=0.1,
        alpha_s=0.4,
        colf_lambda=0.1,
    ):
        super().__init__(agent_id, state_size, action_size, seed, gamma)
        # Let alpha_S > alpha_NS
        self.alpha_ns = alpha_ns
        # arbitrarily picked 4.0
        # if the user doesnt provide alpha_s, use alpha_ns
        self.alpha_s = 4.0 * alpha_ns if alpha_s is None else alpha_s
        # safety check
        if self.alpha_s <= self.alpha_ns:
            raise ValueError("CoLF requires alpha_S > alpha_NS.")
        # store lambda
        self.colf_lambda = colf_lambda

        # initialize: P(a, a_i) <- 0 and S(a, a_i) <- 0 for all a, a_i
        self.P = np.zeros((state_size, action_size))
        self.S = np.zeros((state_size, action_size))

    # Get called after agent takes action and observes reward
    def learn(self, state: int, action: int, reward: float, next_state: int):
        # delta_r_i^t <- | r_i^t - P(a^{t-1}, a_i^t) |
        delta_r = abs(reward - self.P[state, action])

        # if delta_r_i^t > S(a^{t-1}, a_i^t)
        #                  then alpha <- alpha_NS
        #                  else alpha <- alpha_S
        alpha = self.alpha_ns if delta_r > self.S[state, action] else self.alpha_s

        # apply decayed alpha
        alpha = self.get_decayed_alpha(state, action, alpha)

        # Q(a^{t-1}, a_i^t) <- (1-alpha)Q + alpha(r_i^t + gamma max Q(a^t, a_i))
        self.q_table[state, action] = (1 - alpha) * self.q_table[state, action] + alpha * (
            reward + self.gamma * np.max(self.q_table[next_state])
        )

        # S(a^{t-1}, a_i^t) <- (1-lambda)S + lambda * delta_r_i^t
        self.S[state, action] = (1 - self.colf_lambda) * self.S[state, action] + self.colf_lambda * delta_r

        # P(a^{t-1}, a_i^t) <- (1-lambda)P + lambda * r_i^t
        self.P[state, action] = (1 - self.colf_lambda) * self.P[state, action] + self.colf_lambda * reward
