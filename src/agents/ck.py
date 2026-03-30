from .base_agent import BaseAgent

class CKAgent(BaseAgent):
    """
    CK - Change & Keep Algorithm
    """
    def __init__(self, agent_id, state_size, action_size, gamma, base_alpha):
        super().__init__(agent_id, state_size, action_size, gamma, base_alpha)
        self.status = "update"

    def choose_action(self):
        pass

    def learn(self):
        """
        details
        """
        pass