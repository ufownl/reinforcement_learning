import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


class NonstationaryBandit:
    def __init__(self):
        self.__actions = np.zeros(10)

    def __bandit(self, idx):
        self.__actions += np.random.normal(0.0, 0.01, self.__actions.shape)
        return self.__actions[idx]

    def run(self, steps, alpha, epsilon):
        r = np.zeros(steps)
        q = np.zeros(len(self.__actions))
        o = alpha
        for t in range(steps):
            if random.random() < epsilon:
                a = random.randrange(len(self.__actions))
            else:
                a = np.argmax(q)
            r[t] = self.__bandit(a)
            q[a] += (r[t] - q[a]) * alpha / o
            o += alpha * (1 - o)
        return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nonstationary bandit algorithm (Section 2.5).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 2000)", type=int, default=2000)
    parser.add_argument("--steps", help="total number of time-steps (default: 10000)", type=int, default=10000)
    parser.add_argument("--alpha", help="constant step-size parameter (default 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.0625)", type=float, default=0.0625)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.steps, args.alpha, args.epsilon), [NonstationaryBandit() for _ in range(args.rounds)])), 0), label="Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
