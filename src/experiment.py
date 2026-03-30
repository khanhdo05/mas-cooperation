import numpy as np
from .agents.base_agent import BaseAgent

class Experiment:
    """
    Experiment class to run multiple trials of a multi-agent learning scenario.
    """
    def __init__(self, env, agent_class: type[BaseAgent], episodes: int, trials: int, **agent_params):
        self.env = env
        self.agent_class = agent_class
        self.episodes = episodes
        self.trials = trials
        self.agent_params = agent_params

    def _run_single_trial(self):
        """
        Run a single trial of the experiment.
        """
        # Initialize agents and environment for one trial
        agents = [self.agent_class(i, self.env.state_size, self.env.action_space_size, **self.agent_params) 
                  for i in range(self.env.N)]
        state = self.env.reset()
        rewards_history = []

        # Main loop for one trial
        for t in range(self.episodes):
            joint_action = [agent.choose_action(state, t) for agent in agents]
            next_state, rewards, prev_state = self.env.step(joint_action)
            
            # Each agent learn by updating with its own reward and the new state
            for i, agent in enumerate(agents):
                agent.learn(prev_state, joint_action[i], rewards[i], next_state)
            
            state = next_state
            rewards_history.append(np.mean(rewards))
        
        return rewards_history

    def run(self):
        """
        Run the experiment.
        """
        # Store rewards for each trial
        trial_rewards = []

        # Run multiple trials and average the rewards
        for trial in range(self.trials):
            trial_rewards.append(self._run_single_trial())
            print(f"Trial {trial+1}/{self.trials} complete.")

        return np.mean(trial_rewards, axis=0) # Average over all trials