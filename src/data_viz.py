import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_results(results_dict, title, filename):
        plt.figure(figsize=(10, 6))
        i = 0
        for label, data in results_dict.items():
            colors = {
            0: "red",
            1: "green",
            2: "blue",
            3: "magenta"
          }

            # Calculate moving average for smoothing
            window = 1000
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(
                np.arange(len(smoothed))/1000, 
                smoothed, 
                label=label,
                color=colors.get(i, None)
                )
            i += 1
            
        plt.title(title)
        plt.xlabel("Training Episodes (x 1000)")
        plt.ylabel("Average Payoffs")
        plt.ylim(0.3, 1.0)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"results/plots/{filename}.png")
        plt.show()