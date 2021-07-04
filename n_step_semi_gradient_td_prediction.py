import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.random_walk_1000 import State, calc_state_values


def value(state, weight):
    return weight[state.index // 50]


def dvalue(state, weight):
    gradient = np.zeros_like(weight)
    gradient[state.index // 50] = 1
    return gradient


def train(episodes, alpha, steps):
    w = np.zeros(20)
    for _ in range(episodes):
        episode = [(State(), None)]
        t = 0
        while True:
            state, _ = episode[-1]
            if not state is None:
                episode.append(state.transition())
            update_t = t - steps + 1
            if update_t >= 0:
                g = sum([r for _, r in episode[update_t+1:]])
                state, _ = episode[-1]
                if not state is None:
                    g += value(state, w)
                state, _ = episode[update_t]
                w += alpha * (g - value(state, w)) * dvalue(state, w)
                if episode[update_t + 1][0] is None:
                    break
            t += 1
    return np.array([value(State(i), w) for i in range(1000)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="n-step semi-gradient TD Prediction - State aggregation (Example 9.2).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 100)", type=int, default=100)
    parser.add_argument("--episodes", help="number of episodes (default: 10)", type=int, default=10)
    parser.add_argument("--theta", help="accuracy of true value estimation (default: 1e-8)", type=float, default=1e-8)
    args = parser.parse_args()

    true_values = calc_state_values(args.theta)
    x_axis = [0.05 * i for i in range(21)]
    with Pool(cpu_count()) as p:
        for i in range(10):
            steps = 2 ** i
            plt.plot(x_axis, [np.mean(np.sqrt(np.mean((np.concatenate(p.map(Runner(args.episodes, alpha, steps), [train for _ in range(args.rounds)])) - true_values) ** 2, axis=1))) for alpha in x_axis], label="n=%d"%steps)
    plt.grid(True)
    plt.legend()
    plt.show()
