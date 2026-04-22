from src.exp_env.webots_env import WebotsEnv
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.ck_colf import CKCoLFAgent
import numpy as np

def get_algo_name(world_name: str) -> str:
    if "q_learning" in world_name:
        return "Q-Learning"
    elif "ck_colf" in world_name:
        return "CK-CoLF (Hybrid)"
    elif "ck" in world_name:
        return "CK (Change & Keep)"
    elif "colf" in world_name:
        return "CoLF"
    return "Unknown Algorithm"

def make_agent(world_name: str, agent_id: int, env: WebotsEnv):
    fav_num = 31
    seed = np.random.default_rng(fav_num + agent_id)

    if "q_learning" in world_name:
        return QLearningAgent(agent_id, env.n_states, env.n_actions,
                              gamma=0.95, base_alpha=0.1, seed=seed)

    elif "ck" in world_name:
        return CKAgent(agent_id, env.n_states, env.n_actions,
                       gamma=0.95, base_alpha=0.1, seed=seed)

    elif "ck_colf" in world_name:
        return CKCoLFAgent(agent_id, env.n_states, env.n_actions,
                           seed=seed, gamma=0.95,
                           alpha_ns=0.1, alpha_s=0.4,
                           colf_lambda=0.1)
    else:
        raise ValueError(f"Unknown algorithm for agent {agent_id} in file '{world_name}'")
