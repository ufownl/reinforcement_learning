import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils import Runner


class GradientBandit:
    __actions = [
        (0.2, 1),
        (-0.8, 1),
        (1.5, 1),
        (0.5, 1),
        (1.2, 1),
        (-1.5, 1),
        (-0.2, 1),
        (-1, 1),
        (0.8, 1),
        (-0.5, 1)
    ]

    def __bandit(self, idx):
        return random.gauss(*self.__actions[idx])

    def run(self, steps, alpha):
        r = np.zeros(steps)
        q = np.zeros(len(self.__actions))
        n = np.zeros(len(self.__actions))
        h = np.zeros(len(self.__actions))
        for t in range(steps):
            a = np.argmax(h)
            r[t] = self.__bandit(a)
            s = sum([math.exp(h[i]) for i in range(len(self.__actions))])
            p = [math.exp(h[i]) / s for i in range(len(self.__actions))]
            for i in range(len(self.__actions)):
                if i == a:
                    h[i] += alpha * (r[t] - q[i]) * (1 - p[i])
                else:
                    h[i] -= alpha * (r[t] - q[i]) * p[i]
            n[a] += 1
            q[a] += (r[t] - q[a]) / n[a]
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient bandit algorithm (Chapter 2.8).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 2000)", type=int, default=2000)
    parser.add_argument("--steps", help="total number of time-steps (default: 1000)", type=int, default=1000)
    parser.add_argument("--alpha", help="constant step-size parameter (default 0.25)", type=float, default=0.25)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.steps, args.alpha), [GradientBandit() for _ in range(args.rounds)])), 0), label="Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
