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
import json
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller import Supervisor  # type: ignore
from src.webots.utils.helper_functions import get_algo_name, get_communication_data
from src.exp_env.masd_env import MASDEnv
import math
import numpy as np

# =========================
# WEBOTS SETUP
# =========================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
N_AGENTS = 3
for i in range(N_AGENTS):
    print(f"[supervisor] action file {i} = {get_communication_data(i, 'action')}")
    print(f"[supervisor] response file {i} = {get_communication_data(i, 'response')}")

# =========================
# WEBOTS ENVIRONMENT SETUP
# =========================
env = MASDEnv(
    N=N_AGENTS,
    M=3,
    k=2/3
)

state = env.reset()

# =========================
# CAR REFERENCES
# =========================
cars = {
    i: robot.getFromDef(f"car_{i}") for i in range(N_AGENTS)
}

# =========================
# STORE INITIAL POSES
# =========================
initial_poses = {
    i: {
        "translation": cars[i].getField("translation").getSFVec3f(),
        "rotation": cars[i].getField("rotation").getSFRotation(),
    }
    for i in cars
}

# For logging
car_data = {
    i: {"total_dist": 0.0, "prev_pos": None}
    for i in cars
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
MAX_EPISODES = 200000
STEPS_PER_EPISODE = 1000   # Timeout for each attempt
COLLISION_THRESHOLD = 4.0  # Meters (Detects crash between car bounding boxes). If change this, also update in webots_env.py for consistency.
RESET_GRACE_STEPS = 10
grace_counter = 0


# =========================
# MAIN LOOP
# =========================
print(f"=== Starting {MAX_EPISODES} Episodes for {current_algo} ===")

# Episode loop
for episode in range(1, MAX_EPISODES + 1):
    print(f"\n=== Episode {episode} ===", flush=True)
    
    # =========================
    # RESET SHARED DATA
    # =========================
    for i in range(N_AGENTS):
        action_file = get_communication_data(i, 'action')
        response_file = get_communication_data(i, 'response')

        if os.path.exists(action_file):
            os.remove(action_file)

        if os.path.exists(response_file):
            os.remove(response_file)
    time.sleep(0.01)

    # IMPORTANT: allow cars to restart their loop
    for _ in range(5):
        robot.step(timestep)

    # reset episode-specific variables
    state = env.reset()
    episode_step = 0
    collision_occurred = False

    # Step through the episode until timeout or collision
    while robot.step(timestep) != -1:
        episode_step += 1
        # print("[supervisor] checking action files...", flush=True)
        # for i in range(N_AGENTS):
        #     print(i, os.path.exists(get_communication_data(i, 'action')), flush=True)

        # =========================
        # READ ACTIONS FROM CARS
        # =========================
        actions = []
        all_actions_ready = True

        for i in range(N_AGENTS):
            action_file = get_communication_data(i, 'action')

            if not os.path.exists(action_file) or os.path.getsize(action_file) == 0:
                all_actions_ready = False
                break
        # print(
        #     "[supervisor] all_actions_ready =",
        #     all_actions_ready,
        #     flush=True
        # )
        if not all_actions_ready:
            continue

        for i in range(N_AGENTS):
            action_file = get_communication_data(i, 'action')

            try:
                with open(action_file, "r") as f:
                    action_data = json.load(f)
            except json.JSONDecodeError:
                # file exists but not ready yet
                all_actions_ready = False
                break

            actions.append(action_data["action"])
            # print("[supervisor] got actions:", actions, flush=True)
            os.remove(action_file)

        # =========================
        # ENVIRONMENT STEP
        # =========================
        next_state, rewards, prev_state = env.step(actions)
        state = next_state

        # =========================
        # WRITE RESPONSES FOR CARS
        # =========================
        for i in range(N_AGENTS):
            response_file = get_communication_data(i, 'response')

            with open(response_file, "w") as f:
                json.dump({
                    "reward": float(rewards[i]),
                    "next_state": int(next_state)
                }, f)
            # print("[supervisor] wrote responses", flush=True)

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
        if episode_step > 0 and episode_step % 100 == 0:
            print(f"--- Step {episode_step} ---")
            print("ACTIONS:", actions)
            print("STATE:", state)
            print("NEXT STATE:", next_state)
            print("REWARDS:", rewards)
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
                print(f"Epi {episode}/{MAX_EPISODES} | {status} | Avg Spd: {avg_speed:.3f}")

            # =========================
            # FULL RESET
            # =========================

            robot.simulationResetPhysics()
            state = env.reset()

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
                car_data[i]["total_dist"] = 0.0

            break  # Move to next episode

print(f"Training Complete for {current_algo}.")