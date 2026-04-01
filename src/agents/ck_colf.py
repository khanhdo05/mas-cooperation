from .base_agent import BaseAgent
import numpy as np

class CKCoLFAgent(BaseAgent):
    """
    CK-CoLF - Hybrid of Change & Keep and Change or Learn Fast

    This agent combines the "keep" mechanism from CK with the adaptive learning rates from CoLF.
     - When the agent detects a change (like in CK), it enters "keep" mode and uses the stored joint action for updates.
     - When in "keep" mode, it uses the CoLF logic to adapt its learning rate based on the observed reward changes, 
        allowing it to learn quickly when the environment is changing and more slowly when it is stable.
    """
    def __init__(self, agent_id, state_size, action_size, gamma, alpha_ns=0.1, alpha_s=0.4, colf_lambda=0.1):
        super().__init__(agent_id, state_size, action_size, gamma)
        self.alpha_ns = alpha_ns # Learning rate for non-stationarity (how quickly to adapt to changes in the environment)
        self.alpha_s = alpha_s # Learning rate for stationarity (how quickly to learn from stable environments)
        
        # CK mechanism
        self.status = "update"
        # previous action a_i^(t-1)
        self.prev_action = None
        # stored joint action (state)
        self.a_upd = None

        # CoLF mechanism
        self.colf_lambda = colf_lambda

        # initialize: P(a, a_i) <- 0 and S(a, a_i) <- 0 for all a, a_i
        self.P = np.zeros((state_size, action_size))
        self.S = np.zeros((state_size, action_size))

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
        if status == update:
            - if action changed -> go to KEEP mode & store a_upd
            - else -> standard Q update with CoLF adaptive alpha
        else:
            - KEEP mode -> update using stored a_upd with CoLF adaptive alpha   
        """
        if self.status == "update":
            if self.prev_action is not None:
                self.status = "keep"
                self.a_upd = self.prev_action
            else:
                # get CoLF adaptive alpha:
                delta_r = abs(reward - self.P[state, action])
                alpha = self.alpha_ns if delta_r > self.S[state, action] else self.alpha_s
                alpha = self.get_decayed_alpha(state, action, alpha)

                # Standard Q update with CoLF adaptive alpha
                self.q_table[state, action] = (
                    (1 - alpha) * self.q_table[state, action]
                    + alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                )
                
                # S(a^{t-1}, a_i^t) <- (1-lambda)S + lambda * delta_r_i^t
                self.S[state, action] = (1 - self.colf_lambda) * self.S[state, action] + self.colf_lambda * delta_r

                # P(a^{t-1}, a_i^t) <- (1-lambda)P + lambda * r_i^t
                self.P[state, action] = (1 - self.colf_lambda) * self.P[state, action] + self.colf_lambda * reward

        else:
            # KEEP mode -> update using stored a_upd with CoLF adaptive alpha
            if self.a_upd is not None:
                # use stored a_upd for updates instead of current state
                upd_state = self.a_upd if self.a_upd is not None else state

                # get CoLF adaptive alpha:
                delta_r = abs(reward - self.P[upd_state, action])
                alpha = self.alpha_ns if delta_r > self.S[upd_state, action] else self.alpha_s
                alpha = self.get_decayed_alpha(upd_state, action, alpha)

                self.q_table[upd_state, action] = (
                    (1 - alpha) * self.q_table[upd_state, action]
                    + alpha * (reward + self.gamma * np.max(self.q_table[next_state]))
                )

                # S(a^{t-1}, a_i^t) <- (1-lambda)S + lambda * delta_r_i^t
                self.S[upd_state, action] = (1 - self.colf_lambda) * self.S[upd_state, action] + self.colf_lambda * delta_r

                # P(a^{t-1}, a_i^t) <- (1-lambda)P + lambda * r_i^t
                self.P[upd_state, action] = (1 - self.colf_lambda) * self.P[upd_state, action] + self.colf_lambda * reward
            self.status = "update"
        self.prev_action = action   