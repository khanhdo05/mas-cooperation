from src.exp_env.masd_env import MASDEnv
from src.experiment import Experiment
from src.data_viz import DataVisualizer
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.colf import CoLFAgent
from src.agents.ck_colf import CKCoLFAgent

def main():
    # Experimental Setup from Source [1, 2]
    env = MASDEnv(N=3, M=3, k=0.66)
    episodes = 10000
    trials = 20
    viz = DataVisualizer()

    # --- FIGURE 4(a): Q-Learning with Different Learning Rates ---
    print("Simulating Figure 4(a)...")
    alphas = [0.1, 0.2, 0.4, 0.8]
    fig4a_data = {}
    for a in alphas:
        exp = Experiment(env, QLearningAgent, episodes, trials, gamma=0.95, base_alpha=a)
        fig4a_data[f"Q-learning, alpha={a}"] = exp.run()
    viz.plot_results(fig4a_data, "Q-learning: Effect of alpha", "fig_4a")

    # --- FIGURE 4(b): CK with Different Learning Rates ---
    # uncomment the below when implemented 
    # print("Simulating Figure 4(b)...")
    # fig4b_data = {}
    # for a in alphas:
    #     exp = Experiment(env, CKAgent, episodes, trials, gamma=0.95, base_alpha=a)
    #     fig4b_data[f"CK, alpha={a}"] = exp.run()
    # viz.plot_results(fig4b_data, "CK: Effect of alpha", "fig_4b")

    # --- FIGURE 4(c): CoLF vs Q-Learning ---
    # Paper uses alpha_NS=0.1 and alpha_S=0.4 for CoLF [4]
    # uncomment the below when implemented
    # print("Simulating Figure 4(c)...")
    # fig4c_data = {
    #     "Q-learning, alpha=0.1": fig4a_data["Q-learning, alpha=0.1"],
    #     "Q-learning, alpha=0.4": fig4a_data["Q-learning, alpha=0.4"],
    #     "CoLF, alphaNS=0.1, alphaS=0.4": Experiment(env, CoLFAgent, episodes, trials, 
    #                                                 alpha_ns=0.1, alpha_s=0.4).run()
    # }
    # viz.plot_results(fig4c_data, "CoLF vs Q-Learning", "fig_4c")

    # --- FIGURE 4(d): CK-CoLF vs CK ---
    # uncomment the below when implemented
    # print("Simulating Figure 4(d)...")
    # fig4d_data = {
    #     "CK, alpha=0.1": fig4b_data["CK, alpha=0.1"],
    #     "CK, alpha=0.4": fig4b_data["CK, alpha=0.4"],
    #     "CK-CoLF, alphaNS=0.1, alphaS=0.4": Experiment(env, CKCoLFAgent, episodes, trials, 
    #                                                    alpha_ns=0.1, alpha_s=0.4).run()
    # }
    # viz.plot_results(fig4d_data, "CK-CoLF vs CK", "fig_4d")

if __name__ == "__main__":
    main()