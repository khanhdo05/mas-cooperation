from src.exp_env.masd_env import MASDEnv
from src.experiment import Experiment
from src.data_viz import DataVisualizer
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.colf import CoLFAgent
from src.agents.ck_colf import CKCoLFAgent
from src.helper_functions import save_experiment_data, load_experiment_data
import numpy as np

def main():
    # set seed for reproducibility
    np.random.seed(42)
    
    # Experimental Setup
    env = MASDEnv(N=3, M=3, k=float(2/3))
    episodes = 200000
    trials = 100
    viz = DataVisualizer()

    # --- FIGURE 4(a): Q-Learning with Different Learning Rates ---
    print("Simulating Figure 4(a)...")
    alphas = [0.1, 0.2, 0.4, 0.8]
    fig4a_data = {}
    
    for a in alphas:
        filename = f"fig4a_ql_alpha_{a}"
        data = load_experiment_data(filename)
        
        if data is None:
            # If no saved data, run the 100-trial experiment
            exp = Experiment(env, QLearningAgent, episodes, trials, gamma=0.95, base_alpha=a)
            data = exp.run()
            save_experiment_data(data, filename)
            
        fig4a_data[r"Q-learning, $\alpha = {}$".format(a)] = data

    viz.plot_results(fig4a_data, "Q-learning with different learning rates: N=3, M=3", "fig_4a")

    # --- FIGURE 4(b): CK with Different Learning Rates ---
    # uncomment the below when implemented 
    print("Simulating Figure 4(b)...")
    fig4b_data = {}
    for a in alphas:
        filename = f"fig4b_ck_alpha_{a}"
        data = load_experiment_data(filename)

        if data is None:
            # If no saved data, run the 100-trial experiment
            exp = Experiment(env, CKAgent, episodes, trials, gamma=0.95, base_alpha=a)
            data = exp.run()
            save_experiment_data(data, filename)
            
        fig4b_data[r"CK, $\alpha = {}$".format(a)] = data

    viz.plot_results(fig4b_data, "CK with different learning rates: N=3, M=3", "fig_4b")

    # --- FIGURE 4(c): CoLF vs Q-Learning ---
    # uncomment the below when implemented
    print("Simulating Figure 4(c)...")
    colf_filename = "fig4c_colf_ns01_s04"
    colf_data = load_experiment_data(colf_filename)
    
    if colf_data is None:
        # Run CoLF with specific alpha_NS=0.1 and alpha_S=0.4
        exp_colf = Experiment(env, CoLFAgent, episodes, trials, gamma=0.95, alpha_ns=0.1, alpha_s=0.4, colf_lambda=0.1)
        colf_data = exp_colf.run()
        save_experiment_data(colf_data, colf_filename)

    fig4c_data = {
        r"Q-learning, $\alpha = 0.1$": fig4a_data[r"Q-learning, $\alpha = 0.1$"],
        r"Q-learning, $\alpha = 0.4$": fig4a_data[r"Q-learning, $\alpha = 0.4$"],
        r"CoLF, $\alpha_{NS} = 0.1, \alpha_{S} = 0.4$": colf_data
    }

    viz.plot_results(fig4c_data, "CoLF vs Q-Learning: N=3, M=3", "fig_4c")

    # --- FIGURE 4(d): CK-CoLF vs CK ---
    # uncomment the below when implemented
    print("Simulating Figure 4(d)...")

    ck_colf_filename = "fig4d_ck_colf_ns01_s04"
    ck_colf_data = load_experiment_data(ck_colf_filename)
    
    if ck_colf_data is None:
        # Run CK-CoLF with specific alpha_NS=0.1 and alpha_S=0.4
        exp_ck_colf = Experiment(env, CKCoLFAgent, episodes, trials, gamma=0.95, alpha_ns=0.1, alpha_s=0.4, colf_lambda=0.1)
        ck_colf_data = exp_ck_colf.run()
        save_experiment_data(ck_colf_data, ck_colf_filename)

    fig4d_data = {
        r"CK, $\alpha = 0.1$": fig4b_data[r"CK, $\alpha = 0.1$"],
        r"CK, $\alpha = 0.4$": fig4b_data[r"CK, $\alpha = 0.4$"],
        r"CK-CoLF, $\alpha_{NS} = 0.1, \alpha_{S} = 0.4$": ck_colf_data
    }
    viz.plot_results(fig4d_data, "CK-CoLF vs CK: N=3, M=3", "fig_4d")

if __name__ == "__main__":
    main()