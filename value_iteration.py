import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import StateFactory


class State:
    def __init__(self, balance, prob, goal):
        self.balance = balance
        self.actions = [[(balance + a, 1 if balance + a >= goal else 0, prob), (balance - a, 0, 1 - prob)] for a in range(1, min(balance, goal - balance) + 1)]


class GamblerProblem:
    def __init__(self, prob, goal):
        self.values = np.zeros(goal + 1)
        self.policies = np.zeros(goal + 1, dtype=int)
        with Pool(cpu_count()) as p:
            self.__states = p.map(StateFactory(State, prob, goal), range(goal + 1))

    def train(self, theta, max_iters):
        for i in range(max_iters):
            delta = 0
            for s in self.__states:
                if s.actions:
                    v = self.values[s.balance]
                    self.values[s.balance] = max([sum([p * (r + self.values[s1]) for s1, r, p in a]) for a in s.actions])
                    delta = max(delta, abs(v - self.values[s.balance]))
            if delta < theta:
                break
        for s in self.__states:
            if s.actions:
                self.policies[s.balance] = np.argmax([sum([p * (r + self.values[s1]) for s1, r, p in a]) for a in s.actions]) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value Iteration (Example 4.3, Exercise 4.9).")
    parser.add_argument("--prob", help="probability of the coin coming up heads (default: 0.4)", type=float, default=0.4)
    parser.add_argument("--goal", help="gambler's goal that he wants to win (default: 100)", type=int, default=100)
    parser.add_argument("--theta", help="accuracy of estimation (default: 1e-32)", type=float, default=1e-32)
    parser.add_argument("--iters", help="max number of iterations (default: 10000)", type=int, default=10000)
    args = parser.parse_args()

    g = GamblerProblem(args.prob, args.goal)
    g.train(args.theta, args.iters)
    print("Value")
    print(g.values[1:-1])
    print("Policy")
    print(g.policies[1:-1])
    plt.subplot(2, 1, 1)
    plt.plot(g.values[:-1], label="value")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.bar([i for i in range(args.goal)], g.policies[:-1], label="policy")
    plt.grid(True)
    plt.legend()
    plt.show()
