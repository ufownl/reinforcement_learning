import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.random_walk_1000 import State, calc_state_values


class TileCoding:
    def __init__(self, alpha, tilings, width): 
        self.__alpha = alpha / tilings
        dims = 1000 // width + 1
        p = np.array([
            np.arange(dims, dtype=int),
            np.arange(1, dims + 1, dtype=int)
        ]) * width
        offset = width // tilings
        self.__p = np.concatenate([p - i * offset for i in range(tilings)], axis=1)

    @property
    def alpha(self):
        return self.__alpha

    @property
    def initial_weight(self):
        return np.zeros(self.__p.shape[1])

    def feature(self, state):
        return np.logical_and(state.index >= self.__p[0], state.index < self.__p[1]).astype("float")

    def value(self, state, weight):
        return np.matmul(weight, self.feature(state).reshape((self.__p.shape[1], 1))).item()


def train(episodes, basis):
    w = basis.initial_weight
    steps = 0
    d = np.zeros(1000)
    for _ in range(episodes):
        episode = [(State(), None)]
        while True:
            steps += 1
            s, _ = episode[-1]
            if s is None:
                break
            d[s.index] += 1
            episode.append(s.transition())
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            w += basis.alpha * (g - basis.value(episode[t][0], w)) * basis.feature(episode[t][0])
        yield w, d / steps


def sqrt_ve(episodes, basis, true_values):
    return np.array([math.sqrt(sum(u * (true_values - np.array([basis.value(State(i), w) for i in range(1000)])) ** 2)) for w, u in train(episodes, basis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile Coding (Section 9.5.4).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 30)", type=int, default=30)
    parser.add_argument("--episodes", help="number of episodes (default: 5000)", type=int, default=5000)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    true_values = calc_state_values(args.theta)
    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, TileCoding(0.0001, 50, 200), true_values), [sqrt_ve for _ in range(args.rounds)])), axis=0), label="Tile Coding (50 tilings)")
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, TileCoding(0.0001, 1, 200), true_values), [sqrt_ve for _ in range(args.rounds)])), axis=0), label="State aggregation (one tiling)")
    plt.grid(True)
    plt.legend()
    plt.show()
