from .base_agent import BaseAgent
import numpy as np

class CoLFAgent(BaseAgent):
    """
    CoLF - Change or Learn Fast Algorithm.
    """
    def __init__(self, agent_id, state_size, action_size, gamma, alpha_ns=0.1, alpha_s=0.4):
        super().__init__(agent_id, state_size, action_size, gamma)
        self.alpha_ns = alpha_ns # Learning rate for non-stationarity (how quickly to adapt to changes in the environment)
        self.alpha_s = alpha_s # Learning rate for stationarity (how quickly to learn from new information when the environment is stable)
        self.P = np.zeros((state_size, action_size)) # Expected payoff 
        self.S = np.zeros((state_size, action_size)) # Expected variability
       
    def learn(self):
        """
        details
        """
        pass