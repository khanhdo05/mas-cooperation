from controller import Robot  # type: ignore
import json

from src.exp_env.webots_env import WebotsEnv
from src.agents.q_learning import QLearningAgent
from src.agents.ck import CKAgent
from src.agents.ck_colf import CKCoLFAgent

MAX_SPEED = 20.0
TIMESTEP = 32
N_AGENTS = 3

def read_gap(sensor):
    return min(sensor.getValue(), 50.0)

def make_agent(agent_id: int, env: WebotsEnv):
    algo_map = {0: "q", 1: "ck", 2: "ckcolf"}
    algo = algo_map[agent_id]
    if algo == "q":
        return QLearningAgent(agent_id, env.n_states, env.n_actions)
    elif algo == "ck":
        return CKAgent(agent_id, env.n_states, env.n_actions)
    elif algo == "ckcolf":
        return CKCoLFAgent(agent_id, env.n_states, env.n_actions)
    else:
        raise ValueError(f"Unknown algorithm for agent {agent_id}")

def run():
    robot = Robot()
    ts = int(robot.getBasicTimeStep())

    # agent id from name
    agent_id = int(robot.getName().split("_")[1])

    env = WebotsEnv(n_agents=N_AGENTS)
    agent = make_agent(agent_id, env)
    
    motors = [
        robot.getDevice("left_front_wheel"),
        robot.getDevice("right_front_wheel"),
        robot.getDevice("left_rear_wheel"),
        robot.getDevice("right_rear_wheel"),
    ]

    for m in motors:
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    sensor = robot.getDevice("front distance sensor")
    sensor.enable(ts)

    emitter = robot.getDevice("emitter")
    receiver = robot.getDevice("receiver")

    if receiver:
        receiver.setChannel(1)
        receiver.enable(ts)

    received = {}

    while robot.step(ts) != -1:
        gap = read_gap(sensor)
        state = env.get_state(agent_id, gap)

        action = agent.choose_action(state)

        speed = env.action_to_speed(action, base_speed=MAX_SPEED)

        # broadcast
        if emitter:
            emitter.send(json.dumps({
                "id": agent_id,
                "action": action
            }).encode())

        received[agent_id] = action

        # receive others
        if receiver:
            while receiver.getQueueLength() > 0:
                msg = json.loads(receiver.getData().decode())
                received[msg["id"]] = msg["action"]
                receiver.nextPacket()

        # learning
        if len(received) == N_AGENTS:
            rewards = env.step(received)
            reward = rewards[agent_id]

            next_gap = read_gap(sensor)
            next_state = env.get_state(agent_id, next_gap)

            agent.learn(state, action, reward, next_state)

            received = {}

        # DRIVE ALL WHEELS
        for m in motors:
            m.setVelocity(speed)


if __name__ == "__main__":
    run()