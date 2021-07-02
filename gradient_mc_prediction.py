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
    steps = 0
    d = np.zeros(1000)
    for _ in range(episodes):
        episode = [(State(), None)]
        while True:
            steps += 1
            s, _ = episode[-1]
            if s is None:
                break
            d[s.index] += 1
            episode.append(s.transition())
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            w += alpha * (g - value(episode[t][0], w)) * dvalue(episode[t][0], w)
    return w, d / steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Monte-Carlo Prediction - State aggregation (Example 9.1).")
    parser.add_argument("--episodes", help="number of episodes (default: 50000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter (default: 2e-5)", type=float, default=2e-5)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    print("Training...")
    weight, state_dist = train(args.episodes, args.alpha)
    print("Done!")

    plt.subplot(2, 1, 1)
    plt.plot([value(State(i), weight) for i in range(1000)], label="Approximate MC value")
    plt.plot(calc_state_values(args.theta), label="True value")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.bar(range(1000), state_dist, width=1.0)
    plt.show()
