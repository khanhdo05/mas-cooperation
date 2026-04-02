import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    @staticmethod
    def plot_results(results_dict, title, filename):
        plt.figure(figsize=(10, 6))

        ax = plt.gca()

        # Remove top and right spines to match the paper
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Define a color palette for better distinction between lines
        configs = [
            {"color": "red", "marker": "+"},
            {"color": "green", "marker": "x"},
            {"color": "blue", "marker": "*"},
            {"color": "magenta", "marker": "s"} # 's' is the square marker
        ]

        i = 0
        for label, data in results_dict.items():
            config = configs[i % len(configs)]

            # Calculate moving average for smoothing
            window = 1000
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            x_axis = np.arange(len(smoothed)) / 1000
            plt.plot(
                x_axis, 
                smoothed, 
                label=label,
                color=config["color"],
                marker=config["marker"],
                markevery=4000, 
                markersize=5,
                linewidth=1.5
            )
            i += 1
            
        plt.title(title)
        plt.xlabel("Training Episodes (x 1000)")
        plt.ylabel("Average Payoffs")

        plt.xlim(0, 200)
        plt.ylim(0.3, 1.0)

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.xticks(np.arange(0, 201, 20))
        plt.legend(loc='lower right', frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        plt.savefig(f"results/plots/{filename}.png")
        plt.show()