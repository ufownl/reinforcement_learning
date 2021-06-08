import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.maze import init_blocking_mazes, State


def execute_policy(value, epsilon):
    if np.max(value) == np.min(value) or random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def dyna_q(mazes, time_steps, alpha, gamma, epsilon, planning_steps):
    if mazes[0].shape != mazes[1].shape:
        raise ValueError("Invalid mazes")
    maze = mazes[0]
    values = np.zeros(maze.shape + (4,))
    model = {}
    t = 0
    cumulative_reward = 0
    yield cumulative_reward
    while True:
        state = State(np.array(np.where(maze == 1)).transpose().tolist()[0])
        while not state is None:
            action = execute_policy(values[state.index], epsilon)
            index = state.index + (action,)
            s1, r = state.transition(maze, action)
            cumulative_reward += r
            yield cumulative_reward
            values[index] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[index])
            if state.index in model:
                model[state.index][action] = (s1, r)
            else:
                model[state.index] = {action: (s1, r)}
            state = s1
            for _ in range(planning_steps):
                s = random.choice(list(model))
                actions = model[s]
                a = random.choice(list(actions))
                s1, r = actions[a]
                values[s][a] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[s][a])
            t += 1
            if t >= time_steps:
                return
            if t == 1000:
                maze = mazes[1]


def dyna_q_plus(mazes, time_steps, alpha, gamma, epsilon, kappa, planning_steps):
    if mazes[0].shape != mazes[1].shape:
        raise ValueError("Invalid mazes")
    maze = mazes[0]
    values = np.zeros(maze.shape + (4,))
    model = {}
    now = 0
    cumulative_reward = 0
    yield cumulative_reward
    while True:
        state = State(np.array(np.where(maze == 1)).transpose().tolist()[0])
        while not state is None:
            action = execute_policy(values[state.index], epsilon)
            index = state.index + (action,)
            s1, r = state.transition(maze, action)
            cumulative_reward += r
            yield cumulative_reward
            values[index] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[index])
            if state.index in model:
                model[state.index][action] = (s1, r, now)
            else:
                model[state.index] = [(s1, r, now) if a == action else (state, 0, 0) for a in range(4)]
            state = s1
            for _ in range(planning_steps):
                s = random.choice(list(model))
                actions = model[s]
                a = random.randrange(0, len(actions))
                s1, r, t = actions[a]
                values[s][a] += alpha * (r + kappa * math.sqrt(now - t) + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[s][a])
            now += 1
            if now >= time_steps:
                return
            if now == 1000:
                maze = mazes[1]


class CumulativeReward:
    def __init__(self, f):
        self.__f = f

    def __call__(self, *args, **kwargs):
        return np.array([v for v in self.__f(*args, **kwargs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Dyna-Q+ (Example 8.2).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 30)", type=int, default=30)
    parser.add_argument("--time_steps", help="number of time steps (default: 3000)", type=int, default=3000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD (default: 1.0)", type=float, default=1.0)
    parser.add_argument("--gamma", help="discount rate of reward (default: 0.95)", type=float, default=0.95)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--kappa", help="factor of bonus reward (default: 0.001)", type=float, default=0.001)
    parser.add_argument("--planning_steps", help="number of planning steps (default: 50)", type=int, default=50)
    args = parser.parse_args()

    mazes = init_blocking_mazes()
    with Pool(cpu_count()) as p:
        print("Dyna-Q training...")
        plt.plot(np.mean(np.concatenate(p.map(Runner(mazes, args.time_steps, args.alpha, args.gamma, args.epsilon, args.planning_steps), [CumulativeReward(dyna_q) for _ in range(args.rounds)])), 0), label="Dyna-Q")
        print("Dyna-Q+ training...")
        plt.plot(np.mean(np.concatenate(p.map(Runner(mazes, args.time_steps, args.alpha, args.gamma, args.epsilon, args.kappa, args.planning_steps), [CumulativeReward(dyna_q_plus) for _ in range(args.rounds)])), 0), label="Dyna-Q+")
        print("Done!")
    plt.grid(True)
    plt.legend()
    plt.show()
