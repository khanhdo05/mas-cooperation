from .base_agent import BaseAgent

class CKCoLFAgent(BaseAgent):
    def __init__(self, agent_id, state_size, action_size, gamma, base_alpha):
        super().__init__(agent_id, state_size, action_size, gamma, base_alpha)

    def learn(self):
        """
        details
        """
        pass