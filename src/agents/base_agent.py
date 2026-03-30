import numpy as np

class BaseAgent:
    """
    The base class for all agents, already defined `choose_action` for use.
    When adopt, need to implement `learn` and may need to reimplement `choose_action`.
    """
    def __init__(self, agent_id, state_size, action_size, gamma=0.95):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Initialize Q-values for all states and actions to Vmax (optimistic initialization)
        self.v_max = 1.0 / (1.0 - self.gamma) 
        self.q_table = np.full((state_size, action_size), self.v_max)
        
        # Track action counts for learning rate decay
        self.n_table = np.zeros((state_size, action_size), dtype=int)

    def get_epsilon(self, t):
        """
        Decays epsilon linearly from 0.2 to 0.
        """
        return max(0.2 - 0.00006*t, 0)
    
    def choose_action(self, state, t):
        """
        The agent selects an action using the epsilon-greedy policy:
            - With probability epsilon, it chooses a random action (exploration).
            - With probability 1 - epsilon, it chooses the action with the highest Q-value.
        """
        epsilon = self.get_epsilon(t)
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)  # Explore: random action
        else:
            return np.argmax(self.q_table[state])  # Exploit: best action
