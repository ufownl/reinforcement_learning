import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.random_walk_1000 import State, calc_state_values


def value(state, weight):
    return weight[state.index // 100]


def dvalue(state, weight):
    gradient = np.zeros_like(weight)
    gradient[state.index // 100] = 1
    return gradient


def train(episodes, alpha):
    w = np.zeros(10)
    for _ in range(episodes):
        s = State()
        while True:
            s1, r = s.transition()
            if s1 is None:
                w += alpha * (r - value(s, w)) * dvalue(s, w)
                break
            else:
                w += alpha * (r + value(s1, w) - value(s, w)) * dvalue(s, w)
            s = s1
    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-gradient TD(0) Prediction - State aggregation (Example 9.2).")
    parser.add_argument("--episodes", help="number of episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    print("Training...")
    weight = train(args.episodes, args.alpha)
    print("Done!")

    plt.plot([value(State(i), weight) for i in range(1000)], label="Approximate MC value")
    plt.plot(calc_state_values(args.theta), label="True value")
    plt.grid(True)
    plt.legend()
    plt.show()
