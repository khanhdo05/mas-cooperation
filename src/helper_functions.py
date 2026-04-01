import numpy as np
import os

def save_experiment_data(data, filename):
    """Saves the 1D average payoff array to a CSV file."""
    os.makedirs("results/data", exist_ok=True)
    path = os.path.join("results/data", f"{filename}.csv")
    np.savetxt(path, data, delimiter=",")
    print(f"Data saved to {path}")

def load_experiment_data(filename):
    """Loads the 1D average payoff array from a CSV file if it exists."""
    path = os.path.join("results/data", f"{filename}.csv")
    if os.path.exists(path):
        print(f"Loading existing data from {path}...")
        return np.loadtxt(path, delimiter=",")
    return None