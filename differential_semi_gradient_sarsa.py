import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.access_control_queuing import State, Tabular


def train(steps, epsilon, basis):
    w = np.zeros((basis.dimensions, 1))
    avg_r = 0
    s = State()
    a = basis.policy(s, w, epsilon)
    for _ in range(steps):
        x = basis.feature(s, a)
        s1, r = s.transition(a)
        a1 = basis.policy(s1, w, epsilon)
        delta = r - avg_r + basis.value(basis.feature(s1, a1), w) - basis.value(x, w)
        avg_r += basis.beta * delta
        w += basis.alpha * delta * x
        s = s1
        a = a1
    return w, avg_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Semi-gradient Sarsa - Tabular (Example 10.2).")
    parser.add_argument("--steps", help="number of steps (default: 200000000)", type=int, default=200000000)
    parser.add_argument("--alpha", help="constant step-size parameter of weight (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--beta", help="constant step-size parameter of average reward (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()

    basis = Tabular(args.alpha, args.beta)
    print("Training...")
    weight, average_reward = train(args.steps, args.epsilon, basis)
    print("Done!")

    print(np.array([[basis.policy(State(p, n), weight) for n in range(1, 11)] for p in range(4)]))
    print("Average reward: %f" % average_reward)
    for p in range(4):
        plt.plot([max([basis.value(basis.feature(State(p, n), a), weight) for a in range(2)]) for n in range(11)], label="priority %d"%(1<<p))
    plt.grid(True)
    plt.legend()
    plt.show()
