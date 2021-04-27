import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils import Runner


class SimpleBandit:
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

    def run(self, steps, epsilon, q1):
        r = np.zeros(steps)
        q = np.ones(len(self.__actions)) * q1
        n = np.zeros(len(self.__actions))
        for t in range(steps):
            if random.random() < epsilon:
                a = random.randrange(len(self.__actions))
            else:
                a = np.argmax(q)
            r[t] = self.__bandit(a)
            n[a] += 1
            q[a] += (r[t] - q[a]) / n[a]
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple bandit algorithm (Section 2.4, 2.6).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 2000)", type=int, default=2000)
    parser.add_argument("--steps", help="total number of time-steps (default: 1000)", type=int, default=1000)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.0)", type=float, default=0.0)
    parser.add_argument("--q1", help="initial action-value estimates (default: 0.0)", type=float, default=0.0)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.steps, args.epsilon, args.q1), [SimpleBandit() for _ in range(args.rounds)])), 0), label="Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
