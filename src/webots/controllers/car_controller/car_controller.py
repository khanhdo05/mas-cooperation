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
from src.webots.utils.helper_functions import make_agent

MAX_SPEED = 20.0
TIMESTEP = 32
N_AGENTS = 3

# helper function to read gap sensor values and cap them at 50.0 (to avoid outliers)
def read_gap(sensor):
    return min(sensor.getValue(), 50.0)

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
    t = 0
    ts = int(robot.getBasicTimeStep())

    agent_id = int(robot.getName().split("_")[1])

    env = WebotsEnv(n_agents=N_AGENTS)
    world_name = robot.getWorldPath().split('/')[-1]
    agent = make_agent(world_name, agent_id, env)

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
    # COMMUNICATION
    # =========================
    received = {}

    # =========================
    # MAIN LOOP
    # =========================
    while robot.step(ts) != -1:
        obs = [
            sensors["lf"].getValue(),
            sensors["rf"].getValue(),
            sensors["lr"].getValue(),
            sensors["rr"].getValue(),
        ]

        state = env.get_state(agent_id, obs)

        action = agent.choose_action(state, t)
        t += 1
        speed = env.action_to_speed(action, base_speed=MAX_SPEED)

        received[agent_id] = action

        # learning step
        if len(received) == N_AGENTS:
            rewards = env.step(received)

            reward = rewards[agent_id]
            next_obs = [
                sensors["lf"].getValue(),
                sensors["rf"].getValue(),
                sensors["lr"].getValue(),
                sensors["rr"].getValue(),
            ]
            next_state = env.get_state(agent_id, next_obs)

            agent.learn(state, action, reward, next_state)

            received = {}

        # drive all wheels
        for m in motors:
            m.setVelocity(speed)


if __name__ == "__main__":
    run()