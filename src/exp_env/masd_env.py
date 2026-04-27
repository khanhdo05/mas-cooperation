import numpy as np

class MASDEnv:
    """
    Multi-Agent Spatial Dynamics Environment

    This class implements the Multi-Agent Spatial Dynamics (MASD) environment where N agents choose from M actions. 
    The reward structure is designed to encourage cooperation based on a selfishness factor k.
    The state is represented as the previous joint action of all agents, and the action space includes M possible actions for each agent.

    - N: Number of agents
    - M: Number of resource units each agent can hold
    - k: Selfishness factor (e.g., 2/3 means at least 2 out of 3 agents must choose the same action to get a reward)
    """
    def __init__(self, N: int, M: int, k: float):
        self.N = N
        self.M = M
        self.k = k
        self.action_space_size = M + 1  # M actions + 1 for "no action" (if needed)
        self.state_size = self.action_space_size ** N  # State is the previous joint action
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state and return the initial state.
        """
        # Initialize with a random joint action as the first state
        self.current_joint_action = np.random.randint(0, self.M + 1, size=self.N)
        return self.joint_action_to_state(self.current_joint_action)

    def joint_action_to_state(self, joint_action):
        """
        Converts a vector of actions to a unique integer state index.
        """
        state = 0

        # Each agent's action contributes to the state index based on its position and the size of the action space.
        for i, a in enumerate(joint_action):
            # The state index is calculated by treating the joint action as a number in base (M+1), where each agent's action is a digit.
            state += a * (self.action_space_size ** i)
        return state
    
    def step(self, joint_action):
        """
        Calculates rewards based on the MASD utility function.

        Return the next state, rewards for each agent, and the previous state.
        """
        joint_action = np.array(joint_action, dtype=int)
        total_contribution = np.sum(joint_action)  # Total contribution is the sum of all agents' actions. Higher contributions indicate more cooperation.
        avg_contribution = total_contribution / self.N  # Average contribution per agent. This is used to determine the reward based on how much cooperation is happening in the joint action.
        
        rewards = []

        # The reward for each agent is calculated based on the average contribution and the selfishness factor k.
        for i in range(self.N):
            # This is the Payoff Function!
            # The reward is designed to encourage cooperation. If the average contribution is high, agents receive a higher reward.
            reward = (avg_contribution - self.k * joint_action[i]) / (self.M * (1 - self.k))
            rewards.append(reward)
        
        # Update the current joint action and calculate the next state
        prev_state = self.joint_action_to_state(self.current_joint_action)
        self.current_joint_action = np.array(joint_action)
        next_state = self.joint_action_to_state(self.current_joint_action)
        
        return next_state, rewards, prev_state
