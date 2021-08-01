import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.random_walk_1000 import State, calc_state_values


class Polynomial:
    def __init__(self, n, alpha):
        self.__n = n
        self.__alpha = alpha

    @property
    def orders(self):
        return self.__n

    @property
    def alpha(self):
        return self.__alpha

    def feature(self, state):
        return np.array([(state.index / 999) ** i for i in range(self.__n + 1)])

    def value(self, x, weight):
        return np.matmul(weight, x.reshape((self.__n + 1, 1))).item()


class Fourier:
    def __init__(self, n, alpha):
        self.__n = n
        self.__alpha = np.ones(n + 1) * alpha
        self.__alpha[1:] /= np.arange(1, n + 1)

    @property
    def orders(self):
        return self.__n

    @property
    def alpha(self):
        return self.__alpha

    def feature(self, state):
        return np.array([math.cos(math.pi * (state.index / 999) * i) for i in range(self.__n + 1)])

    def value(self, x, weight):
        return np.matmul(weight, x.reshape((self.__n + 1, 1))).item()


def train(episodes, basis):
    w = np.zeros(basis.orders + 1)
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
            x = basis.feature(episode[t][0])
            w += basis.alpha * (g - basis.value(x, w)) * x
        yield w, d / steps


def sqrt_ve(episodes, basis, true_values):
    return np.array([math.sqrt(sum(u * (true_values - np.array([basis.value(basis.feature(State(i)), w) for i in range(1000)])) ** 2)) for w, u in train(episodes, basis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Monte-Carlo Prediction - Polynomial vs Fourier (Section 9.5.1, Section 9.5.2).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 30)", type=int, default=30)
    parser.add_argument("--episodes", help="number of episodes (default: 5000)", type=int, default=5000)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    true_values = calc_state_values(args.theta)
    with Pool(cpu_count()) as p:
        for n in [5, 10, 20]:
            plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, Polynomial(n, 0.0001), true_values), [sqrt_ve for _ in range(args.rounds)])), axis=0), label="Polynomial, n=%d"%n)
            plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, Fourier(n, 0.00005), true_values), [sqrt_ve for _ in range(args.rounds)])), axis=0), label="Fourier, n=%d"%n)
    plt.grid(True)
    plt.legend()
    plt.show()
