import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.mountain_car import State, TileCoding


def train(episodes, epsilon, basis):
    actions = [-1.0, 0.0, 1.0]
    w = np.zeros((basis.dimensions, 1))
    for _ in range(episodes):
        state = State()
        steps = 0
        while True:
            action = basis.policy(state, w, epsilon, actions)
            x = basis.feature(state, action)
            s1, r = state.transition(action)
            steps += 1
            if s1 is None:
                w += basis.alpha * (r - basis.value(x, w)) * x
                break
            v = np.array([basis.value(basis.feature(s1, a), w) for a in actions])
            p = np.ones(len(v)) * epsilon / len(v)
            p[np.argmax(v)] += 1 - epsilon
            w += basis.alpha * (r + np.sum(p * v) - basis.value(x, w)) * x
            state = s1
        yield steps


def episode_steps(episodes, epsilon, basis):
    return np.array([steps for steps in train(episodes, epsilon, basis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-gradient Expected Sarsa - Tile Coding (Exercise 10.2).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 100)", type=int, default=100)
    parser.add_argument("--episodes", help="number of episodes (default: 500)", type=int, default=500)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        for alpha in [0.1, 0.2, 0.5]:
            plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, args.epsilon, TileCoding(alpha)), [episode_steps for _ in range(args.rounds)])), axis=0), label="alpha=%f"%alpha)
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()
