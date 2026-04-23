from src.agents.base_agent import BaseAgent
from src.exp_env.webots_env import WebotsEnv
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.ck_colf import CKCoLFAgent
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../"
    )
)

def get_qtable_path(world_name: str, agent_id: int):
    """
    Creates a stable path:

    mas-cooperation/
        results/
            data/
                q_tables/
                    *.npy
    """
    save_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "data",
        "q_tables"
    )

    os.makedirs(save_dir, exist_ok=True)

    filename = f"{world_name}_car_{agent_id}.npy"

    return os.path.join(save_dir, filename)


def save_q_table(agent: BaseAgent, world_name: str, agent_id: int):
    path = get_qtable_path(world_name, agent_id)

    try:
        np.save(path, agent.q_table)
        print(
            f"[car_{agent_id}] "
            f"Q-table saved -> {path}"
        )

    except Exception as e:
        print(
            f"[car_{agent_id}] "
            f"ERROR saving Q-table: {e}"
        )


def load_q_table(agent: BaseAgent, world_name: str, agent_id: int):
    path = get_qtable_path(world_name, agent_id)

    if not os.path.exists(path):
        print(
            f"[car_{agent_id}] "
            "No saved Q-table found — starting fresh"
        )
        return

    try:
        agent.q_table = np.load(
            path,
            allow_pickle=True
        ).item()

        print(
            f"[car_{agent_id}] "
            f"Loaded Q-table from {path}"
        )

    except Exception as e:
        print(
            f"[car_{agent_id}] "
            f"ERROR loading Q-table: {e}"
        )

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
        return QLearningAgent(agent_id, state_size = env.state_size, action_size = env.action_space_size,
                              gamma=0.95, base_alpha=0.1, seed=seed)
    elif "ck_colf" in world_name:
        return CKCoLFAgent(agent_id, state_size = env.state_size, action_size = env.action_space_size,
                           seed=seed, gamma=0.95,
                           alpha_ns=0.1, alpha_s=0.4,
                           colf_lambda=0.1)
    elif "ck" in world_name:
        return CKAgent(agent_id, state_size = env.state_size, action_size = env.action_space_size,
                       gamma=0.95, base_alpha=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm for agent {agent_id} in file '{world_name}'")
