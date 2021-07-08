import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


def target(x):
    return float(x >= 0.25 and x <= 0.75)


class CoarseCoding:
    def __init__(self, density, width):
        self.__density = density
        p = np.random.uniform(size=density)
        self.__pl = p - width / 2
        self.__pr = p + width / 2

    @property
    def density(self):
        return self.__density

    def alpha(self, a, x):
        n = np.sum(self.feature(x))
        return a / n if n > 1 else a

    def feature(self, x):
        return np.logical_and(x >= self.__pl, x <= self.__pr).astype("float")

    def value(self, x, weight):
        return np.matmul(weight, self.feature(x).reshape((self.__density, 1))).item()


def train(examples, alpha, basis):
    w = np.zeros(basis.density)
    for _ in range(examples):
        x = random.uniform(0.0, 1.0)
        w += basis.alpha(alpha, x) * (target(x) - basis.value(x, w)) * basis.feature(x)
    return w


def approximate(examples, alpha, density, width, x_axis):
    basis = CoarseCoding(density, width)
    w = train(examples, alpha, basis)
    return np.array([basis.value(x, w) for x in x_axis])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coarseness of Coarse Coding (Example 9.3).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 4)", type=int, default=4)
    parser.add_argument("--examples", help="number of examples (default: 10000)", type=int, default=10000)
    parser.add_argument("--alpha", help="constant step-size parameter (default: 0.2)", type=float, default=0.2)
    args = parser.parse_args()

    x_axis = np.arange(101) / 100
    plt.plot(x_axis, [target(x) for x in x_axis], label="Target")
    with Pool(cpu_count()) as p:
        for r, desc in [(0.1, "Narrow"), (0.3, "Medium"), (0.8, "Broad")]:
            plt.plot(x_axis, np.mean(np.concatenate(p.map(Runner(args.examples, args.alpha, 50, r, x_axis), [approximate for _ in range(args.rounds)])), axis=0), label="Approximation, %s"%desc)
    plt.legend()
    plt.show()
