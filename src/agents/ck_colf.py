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
        self.prev_action = None # Previous action a_i^(t-1)
        self.s_upd = None # Suspended state index (at-1)
        self.a_upd = None # Suspended action index (ait)

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
            if self.prev_action is not None and action != self.prev_action:
                self.status = "keep"
                # Suspend the update: store the state-action pair to be updated later
                self.s_upd = state
                self.a_upd = action
            else:
                self._apply_colf_update(state, action, reward, next_state)
        else:
            # KEEP mode: repeat has finished, now update the stored pair
            # Use current reward from the repeated action for the update
            # use stored a_upd for updates instead of current state
            self._apply_colf_update(self.s_upd, self.a_upd, reward, next_state)
            self.status = "update"
        self.prev_action = action   

    def _apply_colf_update(self, s, a, r, s_next):
        """
        Private helper applying the CoLF variable learning rate and 
        the universal alpha decay formula.
        """
        delta_r = abs(r - self.P[s, a])
        alpha_i = self.alpha_ns if delta_r > self.S[s, a] else self.alpha_s
        alpha_t = self.get_decayed_alpha(s, a, alpha_i)
        
        # Standard Q-update
        best_future_q = np.max(self.q_table[s_next])
        self.q_table[s, a] = (1 - alpha_t) * self.q_table[s, a] + alpha_t * (r + self.gamma * best_future_q)
        
        # Update exponential averages for P and S
        self.S[s, a] = (1 - self.colf_lambda) * self.S[s, a] + self.colf_lambda * delta_r
        self.P[s, a] = (1 - self.colf_lambda) * self.P[s, a] + self.colf_lambda * r