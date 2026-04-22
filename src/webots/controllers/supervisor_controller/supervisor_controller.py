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
# STORE INITIAL POSES
# =========================
initial_poses = {}

for i, car in cars.items():
    trans_field = car.getField("translation")
    rot_field = car.getField("rotation")

    initial_poses[i] = {
        "translation": trans_field.getSFVec3f(),
        "rotation": rot_field.getSFRotation(),
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
MAX_STEPS = 200000
STEPS_PER_EPISODE = 1000   # Timeout for each attempt
COLLISION_THRESHOLD = 4.0  # Meters (Detects crash between car bounding boxes). If change this, also update in webots_env.py for consistency.
RESET_GRACE_STEPS = 10
grace_counter = 0

# =========================
# MAIN LOOP
# =========================
print(f"=== Starting {MAX_STEPS} Episodes for {current_algo} ===")

# Episode loop
for episode in range(1, MAX_STEPS + 1):
    # Reset metrics for the new episode
    car_data = {i: {"total_dist": 0.0, "prev_pos": None} for i in cars}  # metrics storage
    episode_step = 0
    collision_occurred = False

    # Step through the episode until timeout or collision
    while robot.step(timestep) != -1:
        episode_step += 1
        
        # ============================
        # UPDATE DISTANCE FOR EACH CAR
        # ============================
        for i, car in cars.items():
            pos = car.getField("translation").getSFVec3f()
            current_pos = (pos[0], pos[1], pos[2])

            if car_data[i]["prev_pos"]:
                prev_x, prev_y, prev_z = car_data[i]["prev_pos"]
                curr_x, curr_y, curr_z = current_pos

                # Calculate Euclidean distance traveled since last step
                # Formula: dist = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
                step_distance = math.sqrt(
                    (curr_x - prev_x) ** 2 +
                    (curr_y - prev_y) ** 2 +
                    (curr_z - prev_z) ** 2
                )

                car_data[i]["total_dist"] += step_distance   

            car_data[i]["prev_pos"] = pos

        # =========================
        # PROGRESS PRINT
        # =========================
        if episode_step % 100 == 0:
            print(f"--- Step {episode_step} ---")
            for i in cars:
                avg_speed = car_data[i]["total_dist"] / episode_step
                print(
                    f"{current_algo} (car_{i}) | "
                    f"distance: {car_data[i]['total_dist']:.3f} | "
                    f"avg speed: {avg_speed:.3f}"
                )

        # ============================
        # COLLISION DETECTION
        # ============================
        if grace_counter > 0:
            grace_counter -= 1
            continue  # Skip collision detection during grace period after reset

        # Check distance between all pairs of cars
        car_positions = {
            i: car_data[i]["prev_pos"]
            for i in cars
        }
        for idx_a in range(len(car_positions)):
            for idx_b in range(idx_a + 1, len(car_positions)):
                pa, pb = car_positions[idx_a], car_positions[idx_b]
                dist_between = math.sqrt(
                    (pa[0]-pb[0]) ** 2 +
                    (pa[1]-pb[1]) ** 2 +
                    (pa[2]-pb[2]) ** 2
                )                
                if dist_between < COLLISION_THRESHOLD:
                    collision_occurred = True
                    break
            if collision_occurred:
                print(f"Collision detected between car_{idx_a} and car_{idx_b} at step {episode_step}!")
                break

        # ============================
        # EPISODE TERMINATION CHECK
        # ============================
        if collision_occurred or episode_step >= STEPS_PER_EPISODE:
            # Log results of the attempt
            avg_speed = sum(m["total_dist"] for m in car_data.values()) / (3 * episode_step)
            
            if episode % 500 == 0: # Status update every 500 episodes
                status = "CRASH" if collision_occurred else "TIMEOUT"
                print(f"Epi {episode}/{MAX_STEPS} | {status} | Avg Spd: {avg_speed:.3f}")

            # =========================
            # FULL RESET
            # =========================

            robot.simulationResetPhysics()

            # Reset positions
            for i, car in cars.items():
                car.getField("translation").setSFVec3f(
                    initial_poses[i]["translation"]
                )

                car.getField("rotation").setSFRotation(
                    initial_poses[i]["rotation"]
                )

            # Let physics settle
            for _ in range(5):
                robot.step(timestep)

            grace_counter = RESET_GRACE_STEPS

            # Reset previous position for distance calculation
            for i in car_data:
                car_data[i]["prev_pos"] = None  

            break  # Move to next episode

print(f"Training Complete for {current_algo}.")