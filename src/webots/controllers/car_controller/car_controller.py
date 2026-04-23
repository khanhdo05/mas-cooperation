import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from controller import Robot  # type: ignore

from src.webots.utils.helper_functions import make_agent, load_q_table, save_q_table

# =========================
# CONSTANTS
# =========================
MAX_SPEED = 13.0
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
    world_name = robot.getWorldPath().split('/')[-1]
    agent = make_agent(world_name, agent_id, state_size=N**M, action_size=M)
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
    state = 0  # Initial state (can be modified to reflect actual environment state if needed)

    while robot.step(ts) != -1:
        # choose action
        action = agent.choose_action(state, t)

        # TODO: get reward and next_state from environment based on action
        # this is a placeholder and should be replaced with actual logic to interact with the environment and calculate rewards
        reward = 0
        next_state = state
        agent.learn(
            state,
            action,
            reward,
            next_state
        )
        state = next_state

        # get speed from action
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