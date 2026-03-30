from .base_agent import BaseAgent

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
        # stored joint action a^upd from the algorithm
        self.a_upd = None
    
    def get_decayed_alpha(self, state, action):
        """Calculates decayed alpha based on the frequency of action in state"""
        # Increment the count for this state-action pair
        self.n_table[state, action] += 1

        # Decay alpha based on the count of how many times this state-action pair has been taken. 
        # The more it has been taken, the smaller the learning rate, allowing for more stable learning over time.
        return self.base_alpha / (1 + 0.0001 * self.n_table[state, action])

    def choose_action(self):
        pass

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        details
        """
        pass