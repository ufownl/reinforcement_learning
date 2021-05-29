import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


class StateB:
    @property
    def index(self):
        return 1

    def transition(self, action):
        return (None, random.gauss(-0.1, 1))


class StateA:
    @property
    def index(self):
        return 0

    def transition(self, action):
        return (StateB(), 0) if action == 0 else (None, 0)


def execute_policy(value, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def q_learning(episodes, alpha, epsilon):
    values = [[random.gauss(0, 1) for _ in range(2)], [random.gauss(0, 1) for _ in range(10)]]
    for _ in range(episodes):
        state = StateA()
        while True:
            action = execute_policy(values[state.index], epsilon)
            if type(state) is StateA:
                yield action == 0
            s1, r = state.transition(action)
            if s1 is None:
                values[state.index][action] += alpha * (r - values[state.index][action])
                break
            values[state.index][action] += alpha * (r + max(values[s1.index]) - values[state.index][action])
            state = s1


def double_q_learning(episodes, alpha, epsilon):
    values = [[[random.gauss(0, 1) for _ in range(2)], [random.gauss(0, 1) for _ in range(10)]] for _ in range(2)]
    for _ in range(episodes):
        state = StateA()
        while True:
            action = execute_policy([sum(v) for v in zip(values[0][state.index], values[1][state.index])], epsilon)
            if type(state) is StateA:
                yield action == 0
            s1, r = state.transition(action)
            qi = random.random() < 0.5
            if s1 is None:
                values[qi][state.index][action] += alpha * (r - values[qi][state.index][action])
                break
            values[qi][state.index][action] += alpha * (r + values[not qi][s1.index][np.argmax(values[qi][s1.index])] - values[qi][state.index][action])
            state = s1


class MaximizationBias:
    def __init__(self, f):
        self.__f = f

    def run(self, episodes, alpha, epsilon):
        return np.array([v for v in self.__f(episodes, alpha, epsilon)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double Q-learning (Example 6.7)")
    parser.add_argument("--rounds", help="total number of running rounds (default: 10000)", type=int, default=10000)
    parser.add_argument("--episodes", help="number of training episodes (default: 1000)", type=int, default=1000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        print("Q-learning training...")
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, args.alpha, args.epsilon), [MaximizationBias(q_learning) for _ in range(args.rounds)])), 0), label="Q-learning")
        print("Double Q-learning training...")
        plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, args.alpha, args.epsilon), [MaximizationBias(double_q_learning) for _ in range(args.rounds)])), 0), label="Double Q-learning")
        print("Done!")
    plt.grid(True)
    plt.legend()
    plt.show()
