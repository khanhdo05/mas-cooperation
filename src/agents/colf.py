from .base_agent import BaseAgent
import numpy as np

class CoLFAgent(BaseAgent):
    """
    CoLF - Change or Learn Fast Algorithm.
    """
    def __init__(self, agent_id, state_size, action_size, gamma, base_alpha):
        super().__init__(agent_id, state_size, action_size, gamma, base_alpha)
        self.P = np.zeros((state_size, action_size)) # Expected payoff 
        self.S = np.zeros((state_size, action_size)) # Expected variability
       
    def learn(self):
        """
        details
        """
        pass