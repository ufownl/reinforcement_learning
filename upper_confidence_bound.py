import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


class UCBBandit:
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

    def __call__(self, steps, confidence):
        r = np.zeros(steps)
        q = np.zeros(len(self.__actions))
        n = np.zeros(len(self.__actions))
        for t in range(steps):
            a = np.argmin(n)
            if n[a] > 0:
                a = np.argmax(q + confidence * np.sqrt(np.ones_like(n) * math.log(t) / n))
            r[t] = self.__bandit(a)
            n[a] += 1
            q[a] += (r[t] - q[a]) / n[a]
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upper-Confidence-Bound action selection (Section 2.7).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 2000)", type=int, default=2000)
    parser.add_argument("--steps", help="total number of time-steps (default: 1000)", type=int, default=1000)
    parser.add_argument("--confidence", help="confidence level the measure of uncertainty (default: 1.0)", type=float, default=1.0)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.steps, args.confidence), [UCBBandit() for _ in range(args.rounds)])), 0), label="Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
