import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.random_walk_1000 import State, calc_state_values


class TileCoding:
    def __init__(self, tilings, width): 
        dims = 1000 // width + 1
        p = np.array([
            np.arange(dims, dtype=int),
            np.arange(1, dims + 1, dtype=int)
        ]) * width
        offset = width // tilings
        self.__p = np.concatenate([p - i * offset for i in range(tilings)], axis=1)

    @property
    def dimensions(self):
        return self.__p.shape[1]

    def feature(self, state):
        if state is None:
            return np.zeros((self.dimensions, 1))
        else:
            return np.logical_and(state.index >= self.__p[0], state.index < self.__p[1]).reshape((self.dimensions, 1)).astype("float")

    def value(self, state, weight):
        return np.matmul(np.transpose(weight), self.feature(state)).item()


def train(episodes, epsilon, basis):
    a = epsilon ** -1 * np.identity(basis.dimensions)
    b = np.zeros((basis.dimensions, 1))
    for _ in range(episodes):
        s = State()
        x = basis.feature(s)
        while not s is None:
            s1, r = s.transition()
            x1 = basis.feature(s1)
            v = np.matmul(np.transpose(a), x - x1)
            a -= np.matmul(np.matmul(a, x), np.transpose(v)) / (1 + np.matmul(np.transpose(v), x))
            b += r * x
            s = s1
            x = x1
    return np.matmul(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Least-Squares TD Prediction - Tile Coding (Section 9.8).")
    parser.add_argument("--episodes", help="number of episodes (default: 10000)", type=int, default=10000)
    parser.add_argument("--epsilon", help="factor of the initial identity matrix (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    basis = TileCoding(50, 200)
    print("Training...")
    weight = train(args.episodes, args.epsilon, basis)
    print("Done!")

    plt.plot([basis.value(State(i), weight) for i in range(1000)], label="Approximate MC value")
    plt.plot(calc_state_values(args.theta), label="True value")
    plt.grid(True)
    plt.legend()
    plt.show()
