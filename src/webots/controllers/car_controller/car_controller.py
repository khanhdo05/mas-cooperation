import sys
import os
import json
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from controller import Robot  # type: ignore

from src.webots.utils.helper_functions import make_agent, load_q_table, save_q_table, get_communication_data

# =========================
# CONSTANTS
# =========================
MAX_SPEED = 20.0
N = 3
M = 3
k = 2/3

def safe_get(robot, name, kind="device"):
    """
    Safely get a device from the robot. Raises an error if the device is missing.
    """
    obj = robot.getDevice(name)
    if obj is None:
        raise RuntimeError(f"[ERROR] Missing {kind}: '{name}'")
    return obj

def action_to_speed(action):
    """
    Convert action index to speed value.
    """
    return MAX_SPEED * (
            1 - action / M
        )

def run():
    robot = Robot()

    ts = int(robot.getBasicTimeStep())

    agent_id = int(robot.getName().split("_")[1])
    ACTION_FILE = get_communication_data(agent_id, 'action')
    RESPONSE_FILE = get_communication_data(agent_id, 'response')
    print(f"[car_{agent_id}] ACTION_FILE = {ACTION_FILE}")
    print(f"[car_{agent_id}] RESPONSE_FILE = {RESPONSE_FILE}")
    world_name = robot.getWorldPath().split('/')[-1]
    agent = make_agent(world_name, agent_id, state_size=(M + 1) ** N, action_size=M+1)
    load_q_table(agent, world_name, agent_id)

    # =========================
    # TESLA MODEL 3 WHEELS
    # =========================
    motors = [
        safe_get(robot, "left_rear_wheel"),
        safe_get(robot, "right_rear_wheel"),
    ]

    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    # =========================
    # MAIN LOOP
    # =========================
    t = 0
    state = 0
    waiting_for_response = False
    pending_action = None

    while robot.step(ts) != -1:
        # =========================
        # IF WAITING, CHECK FOR RESPONSE
        # =========================
        if waiting_for_response:
            if os.path.exists(RESPONSE_FILE):
                with open(RESPONSE_FILE, "r") as f:
                    data = json.load(f)

                reward = data["reward"]
                next_state = data["next_state"]

                os.remove(RESPONSE_FILE)

                agent.learn(
                    state,
                    pending_action,
                    reward,
                    next_state
                )
                state = next_state
                waiting_for_response = False
                pending_action = None

            # keep current wheel speed while waiting
            continue

        # =========================
        # CHOOSE ACTION
        # =========================
        action = agent.choose_action(state, t)

        # =========================
        # SEND ACTION
        # =========================
        tmp = ACTION_FILE + ".tmp"

        with open(tmp, "w") as f:
            json.dump({
                "state": int(state),
                "action": int(action)
            }, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, ACTION_FILE)

        print(f"[car_{agent_id}] wrote action file", flush=True)

        pending_action = action
        waiting_for_response = True

        # =========================
        # APPLY SPEED
        # =========================
        speed = action_to_speed(action)

        # drive all wheels
        for m in motors:
            m.setVelocity(speed)

        # increment time step
        t += 1

        # periodic persistence
        if t % 50 == 0:
            save_q_table(agent, world_name, agent_id)


if __name__ == "__main__":
    run()