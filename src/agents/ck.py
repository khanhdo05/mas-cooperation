from .base_agent import BaseAgent
import numpy as np

class CKAgent(BaseAgent):
    """
    CK - Change & Keep Algorithm
    """
    def __init__(self, agent_id, state_size, action_size, gamma=0.95, base_alpha=0.1):
        super().__init__(agent_id, state_size, action_size, gamma)
        self.base_alpha = base_alpha
        self.status = "update"
        
        # previous action a_i^(t-1)
        self.prev_action = None
        # stored joint action (state)
        self.a_upd = None
    
    def get_decayed_alpha(self, state, action):
        """Calculates decayed alpha based on the frequency of action in state"""
        # Increment the count for this state-action pair
        self.n_table[state, action] += 1

        # Decay alpha based on the count of how many times this state-action pair has been taken. 
        # The more it has been taken, the smaller the learning rate, allowing for more stable learning over time.
        return self.base_alpha / (1 + 0.0001 * self.n_table[state, action])

    def choose_action(self, state, t):
        """
        If status == keep -> repeat last action
        Else -> epsilon-greedy (same as BaseAgent)
        """
        if self.status == "keep" and self.prev_action is not None:
            return self.prev_action
        return super().choose_action(state, t)

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        state      = a^(t-1) (joint action)
        action     = a_i^t
        next_state = a^t
        """
        if self.status == "update":
            # if action changed -> go to KEEP mode
            if self.prev_action is not None and action != self.prev_action:
                self.status = "keep"
                self.a_upd = state
            else:
                # standard Q update
                alpha = self.get_decayed_alpha(state, action)
                self.q_table[state, action] = (
                    (1 - alpha) * self.q_table[state, action]
                    + alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                )
        else:
            # KEEP mode -> update using stored a_upd
            upd_state = self.a_upd if self.a_upd is not None else state
            alpha = self.get_decayed_alpha(upd_state, action)
            self.q_table[upd_state, action] = (
                (1 - alpha) * self.q_table[upd_state, action]
                + alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
            )
            self.status = "update"
        self.prev_action = action