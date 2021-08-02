import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.mountain_car import State, TileCoding


def train(episodes, epsilon, steps, basis):
    w = np.zeros((basis.dimensions, 1))
    for _ in range(episodes):
        state = State()
        episode = [(state, None, basis.policy(state, w, epsilon))]
        t = 0
        while True:
            state, _, action = episode[-1]
            if not state is None:
                s1, r = state.transition(action)
                episode.append((s1, r, None if s1 is None else basis.policy(s1, w, epsilon)))
            update_t = t + 1 - steps
            if update_t >= 0:
                g = sum([r for _, r, _ in episode[update_t+1:]])
                state, _, action = episode[-1]
                if not state is None:
                    g += basis.value(basis.feature(state, action), w)
                state, _, action = episode[update_t]
                x = basis.feature(state, action)
                w += basis.alpha * (g - basis.value(x, w)) * x
                if episode[update_t + 1][0] is None:
                    break
            t += 1
        yield len(episode)


def episode_steps(episodes, epsilon, steps, basis):
    return np.array([n for n in train(episodes, epsilon, steps, basis)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-gradient n-step Sarsa - Tile Coding (Section 10.2).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 100)", type=int, default=100)
    parser.add_argument("--episodes", help="number of episodes (default: 500)", type=int, default=500)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.0)", type=float, default=0.0)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        for steps, alpha in [(1, 0.5), (8, 0.3)]:
            plt.plot(np.mean(np.concatenate(p.map(Runner(args.episodes, args.epsilon, steps, TileCoding(alpha)), [episode_steps for _ in range(args.rounds)])), axis=0), label="n=%d, alpha=%f"%(steps,alpha))
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()
