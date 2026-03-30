import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_results(results_dict, title="MASD Performance"):
        plt.figure(figsize=(10, 6))
        for label, data in results_dict.items():
            # Calculate moving average for smoothing
            window = 1000
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(np.arange(len(smoothed))/1000, smoothed, label=label)
            
        plt.xlabel("Training Episodes (x 1000)")
        plt.ylabel("Average Payoffs")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig("results/plots/reproduction_fig4.png")
        plt.show()