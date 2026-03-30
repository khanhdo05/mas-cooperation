from .base_agent import BaseAgent
import numpy as np

class QLearningAgent(BaseAgent):
    """
    Q-Learning Baseline Algorithm
    """
    def __init__(self, agent_id, state_size, action_size, gamma, base_alpha):
        super().__init__(agent_id, state_size, action_size, gamma, base_alpha)

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        Bellman equation for Q-learning:
        Q(state, action) = Q(state, action) + alpha * (reward + gamma * max(Q(next_state)) - Q(state, action))
        """
        alpha = self.get_decayed_alpha(state, action)

        self.q_table[state, action] += alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])