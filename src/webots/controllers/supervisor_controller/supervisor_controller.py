# =========================
# SUPERVISOR PURPOSE
# =========================
# This Supervisor controller monitors all cars in a self-play simulation.
# It identifies the shared algorithm from the world name and evaluates
# collective performance (distance and speed) across the group.
#
# Specifically, it computes:
# - total distance traveled by each car
# - average speed over time
# =========================

import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from controller import Supervisor  # type: ignore
from src.webots.utils.helper_functions import get_algo_name
import math

# =========================
# WEBOTS SETUP
# =========================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# =========================
# CAR REFERENCES
# =========================
cars = {
    0: robot.getFromDef("car_0"),
    1: robot.getFromDef("car_1"),
    2: robot.getFromDef("car_2"),
}

# =========================
# ALGORITHM MATCHING
# =========================
world_path = robot.getWorldPath()
world_file = os.path.basename(world_path)
current_algo = get_algo_name(world_file)

# =========================
# EXPERIMENT SETTINGS
# =========================
MAX_STEPS = 500

# =========================
# METRIC STORAGE
# =========================
car_data = {
    i: {
        "prev_pos": None,
        "total_distance": 0.0,
    }
    for i in cars
}

print(f"Supervisor started for Experiment: {current_algo}")

# =========================
# MAIN LOOP
# =========================
step_count = 0

while robot.step(timestep) != -1:
    step_count += 1

# =========================
# UPDATE DISTANCE FOR EACH CAR
# =========================
    for i, car in cars.items():
        if car is None:
            continue

        pos = car.getField("translation").getSFVec3f()
        current_pos = (pos[0], pos[1], pos[2])

        if car_data[i]["prev_pos"] is not None:
            prev_x, prev_y, prev_z = car_data[i]["prev_pos"]
            curr_x, curr_y, curr_z = current_pos

            step_distance = math.sqrt(
                (curr_x - prev_x) ** 2 +
                (curr_y - prev_y) ** 2 +
                (curr_z - prev_z) ** 2
            )

            car_data[i]["total_distance"] += step_distance

        car_data[i]["prev_pos"] = current_pos

# =========================
# PROGRESS PRINT
# =========================
    if step_count % 100 == 0:
        print(f"--- Step {step_count} ---")
        for i in cars:
            avg_speed = car_data[i]["total_distance"] / step_count
            print(
                f"{current_algo} (car_{i}) | "
                f"distance: {car_data[i]['total_distance']:.3f} | "
                f"avg speed: {avg_speed:.3f}"
            )

# =========================
# END EXPERIMENT AND PRINT FINAL RESULTS
# =========================
    if step_count >= MAX_STEPS:
        print("\n=== FINAL RESULTS ===")
        for i in cars:
            total_distance = car_data[i]["total_distance"]
            avg_speed = total_distance / step_count

            print(
                f"{current_algo} (car_{i}) | "
                f"total distance: {total_distance:.3f} | "
                f"average speed: {avg_speed:.3f}"
            )
        break