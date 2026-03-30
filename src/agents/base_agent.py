import numpy as np

class BaseAgent:
    """
    The base class for all agents, already defined `choose_action` for use.
    When adopt, need to implement `learn` and may need to reimplement `choose_action`.
    """
    def __init__(self, agent_id, state_size, action_size, gamma=0.95, base_alpha=0.1):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.base_alpha = base_alpha
        
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
        
    def get_decayed_alpha(self, state, action):
        """Calculates decayed alpha based on the frequency of action in state"""
        # Increment the count for this state-action pair
        self.n_table[state, action] += 1

        # Decay alpha based on the count of how many times this state-action pair has been taken. 
        # The more it has been taken, the smaller the learning rate, allowing for more stable learning over time.
        return self.base_alpha / (1 + 0.0001 * self.n_table[state, action])
