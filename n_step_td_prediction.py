import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


class State:
    true_values = np.array([i / 20 for i in range(1, 20)])

    def __init__(self, index=9):
        if index < 0 or index > 18:
            raise IndexError("Invalid state")
        self.__index = index

    @property
    def index(self):
        return self.__index

    def transition(self):
        index = self.__index + random.choice([-1, 1])
        if index < 0:
            return (None, 0)
        elif index > 18:
            return (None, 1)
        else:
            return (State(index), 0)


def train(episodes, alpha, steps):
    values = np.zeros(19)
    for _ in range(episodes):
        episode = [(State(), None)]
        terminal_t = float("inf")
        t = 0
        while True:
            if t < terminal_t:
                episode.append(episode[-1][0].transition())
                if episode[-1][0] is None:
                    terminal_t = t + 1
            update_t = t - steps + 1
            if update_t >= 0:
                g = sum([episode[i][1] for i in range(update_t + 1, min(update_t + steps, terminal_t) + 1)])
                if update_t + steps < terminal_t:
                    g += values[episode[update_t + steps][0].index]
                values[episode[update_t][0].index] += alpha * (g - values[episode[update_t][0].index])
            if update_t == terminal_t - 1:
                break
            t += 1
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-step TD prediction (Example 7.1).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 200)", type=int, default=200)
    parser.add_argument("--episodes", help="number of episodes (default: 10)", type=int, default=10)
    args = parser.parse_args()

    x_axis = [0.05 * i for i in range(21)]
    with Pool(cpu_count()) as p:
        for i in range(10):
            steps = 2 ** i
            plt.plot(x_axis, [np.mean(np.sqrt(np.mean((np.concatenate(p.map(Runner(args.episodes, alpha, steps), [train for _ in range(args.rounds)])) - State.true_values) ** 2, axis=1))) for alpha in x_axis], label="n=%d"%steps)
    plt.grid(True)
    plt.legend()
    plt.show()
