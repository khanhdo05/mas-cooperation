import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from controller import Robot  # type: ignore

from src.exp_env.webots_env import WebotsEnv
from src.webots.utils.helper_functions import make_agent, load_q_table, save_q_table

# =========================
# CONSTANTS
# =========================
MAX_SPEED = 13.0
N = 3
M = 3
k = 2/3

def safe_get(robot, name, kind="device"):
    obj = robot.getDevice(name)
    if obj is None:
        raise RuntimeError(f"[ERROR] Missing {kind}: '{name}'")
    return obj

def run():
    robot = Robot()

    # print("=== DEVICES ===")
    # for i in range(robot.getNumberOfDevices()):
    #     print(robot.getDeviceByIndex(i).getName())
    """
    left_steer 
    left_steer_sensor 
    left_front_sensor 
    left_front_brake 
    right_steer 
    right_steer_sensor 
    right_front_sensor 
    right_front_brake 
    left_rear_wheel 
    left_rear_sensor 
    left_rear_brake 
    right_rear_wheel 
    right_rear_sensor 
    right_rear_brake 
    engine_speaker 
    front_lights 
    right_indicators 
    left_indicators 
    antifog_lights 
    brake_lights 
    rear_lights 
    backwards_lights
    """
    ts = int(robot.getBasicTimeStep())

    agent_id = int(robot.getName().split("_")[1])

    env = WebotsEnv(N, M, k, MAX_SPEED)
    world_name = robot.getWorldPath().split('/')[-1]
    agent = make_agent(world_name, agent_id, env)
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
    # SENSOR
    # =========================
    sensors = {
        "lf": robot.getDevice("left_front_sensor"),
        "rf": robot.getDevice("right_front_sensor"),
        "lr": robot.getDevice("left_rear_sensor"),
        "rr": robot.getDevice("right_rear_sensor"),
    }

    for s in sensors.values():
        s.enable(ts)

    # =========================
    # MAIN LOOP
    # =========================
    state = env.reset()

    t = 0

    while robot.step(ts) != -1:
        action = agent.choose_action(state, t)
        t += 1
        speed = env.action_to_speed(action, max_speed=MAX_SPEED)

        # learning step
        joint_action = np.zeros(env.N, dtype=int)
        joint_action[agent_id] = action

        next_state, rewards, prev_state = env.step(joint_action)
        reward = rewards[agent_id]

        agent.learn(state, action, reward, next_state)

        state = next_state

        # drive all wheels
        for m in motors:
            m.setVelocity(speed)

        # periodic persistence
        if t % 50 == 0:
            print(agent.q_table)
            save_q_table(agent, world_name, agent_id)


if __name__ == "__main__":
    run()