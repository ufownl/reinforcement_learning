import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, index=None):
        if index is None:
            self.__idx = random.randrange(7)
        else:
            self.__idx = index

    @property
    def index(self):
        return self.__idx

    def transition(self, action):
        if action == 0:
            return State(random.randrange(6)), 0
        else:
            return State(6), 0


class StateCoding:
    __x = np.array([
        [2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 2]
    ])

    @property
    def dimensions(self):
        return self.__x.shape[0]

    def feature(self, state):
        return self.__x[:, state.index].reshape((-1, 1))

    def value(self, x, weight):
        return np.matmul(np.transpose(weight), x).item()


def train(steps, alpha, gamma, basis):
    w = np.random.random((2, basis.dimensions, 1))
    state = State()
    for _ in range(steps):
        x = basis.feature(state)
        action = 0 if random.randrange(7) < 6 else 1
        s1, r = state.transition(action)
        x1 = basis.feature(s1)
        w[action] += alpha * (r + gamma * max([basis.value(x1, w[a]) for a in range(2)]) - basis.value(x, w[action])) * x
        state = s1
        yield np.amax(w, axis=0).reshape((-1,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-gradient Q-learning on Baird’s counterexample (Exercise 11.3).")
    parser.add_argument("--steps", help="number of steps (default: 10000)", type=int, default=10000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--gamma", help="discount rate parameter of TD(0) (default: 0.99)", type=float, default=0.99)
    args = parser.parse_args()

    weight = np.concatenate([np.expand_dims(w, 0) for w in train(args.steps, args.alpha, args.gamma, StateCoding())])
    for i in range(weight.shape[1]):
        plt.plot(weight[:, i], label="w%d"%i)
    plt.grid(True)
    plt.legend()
    plt.show()
