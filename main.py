from src.exp_env.masd_env import MASDEnv
from src.experiment import Experiment
from src.data_viz import DataVisualizer
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.colf import CoLFAgent
from src.agents.ck_colf import CKCoLFAgent

def main():
    # Init environment for Figure 4 specifically
    env = MASDEnv(N = 3, M = 3, k = 2/3)

    # Configure episodes and trials
    E = 200000
    T = 100

    # ------------- Q-Learning Agent -------------
    print("Running Q-Learning...")
    exp_ql = Experiment(env, QLearningAgent, episodes=E, trials=T, gamma=0.95, base_alpha=0.1)
    ql_results = exp_ql.run()

    # ------------- CoLF Agent -------------
    print("Running CoLFAgent...")
    exp_colf = Experiment(env, CoLFAgent, episodes=E, trials=T, gamma=0.1, base_alpha=0.1)
    colf_results = exp_colf.run()

    # ------------- CK Agent -------------
    print("Running CKAgent...")
    exp_ck = Experiment(env, CKAgent, episodes=E, trials=T, gamma=0.95, base_alpha=0.1)
    ck_results = exp_ck.run()
    
    # ------------- CK-CoLF Agent -------------
    print("Running Hybrid CK-CoLF...")
    exp_ck_colf = Experiment(env, CKCoLFAgent, episodes=E, trials=T, gamma=0.95, base_alpha=0.1)
    ck_colf_results = exp_ck_colf.run()
    
    results_map = {
        "Q-Learning (alpha=0.1)": ql_results,
        "CoLF (alpha=0.1)": colf_results,
        "CK (alpha=0.1)": ck_results,
        "CK-CoLF (alpha=0.1)": ck_colf_results,
    }
    
    DataVisualizer.plot_results(results_map, title="Reproduction of Figure 4")

if __name__ == "__main__":
    main()