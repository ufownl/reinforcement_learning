import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.access_control_queuing import State, Tabular


def train(steps, epsilon, n, basis):
    w = np.zeros((basis.dimensions, 1))
    avg_r = 0
    state = State()
    queue = [(state, None, basis.policy(state, w, epsilon))]
    for t in range(steps):
        state, _, action = queue[-1]
        s1, r = state.transition(action)
        queue.append((s1, r, basis.policy(s1, w, epsilon)))
        update_t = t + 1 - n
        if update_t >= 0:
            queue = queue[-n-1:]
            delta = sum([r - avg_r for _, r, _ in queue[1:]])
            state, _, action = queue[-1]
            delta += basis.value(basis.feature(state, action), w)
            state, _, action = queue[0]
            x = basis.feature(state, action)
            delta -= basis.value(x, w)
            avg_r += basis.beta * delta
            w += basis.alpha * delta * x
    return w, avg_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Semi-gradient n-step Sarsa - Tabular (Section 10.5, Example 10.2).")
    parser.add_argument("--steps", help="number of steps (default: 200000000)", type=int, default=200000000)
    parser.add_argument("--alpha", help="constant step-size parameter of weight (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--beta", help="constant step-size parameter of average reward (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--n", help="number of steps before bootstrapping (default: 4)", type=int, default=4)
    args = parser.parse_args()

    basis = Tabular(args.alpha, args.beta)
    print("Training...")
    weight, average_reward = train(args.steps, args.epsilon, args.n, basis)
    print("Done!")

    print(np.array([[basis.policy(State(p, n), weight) for n in range(1, 11)] for p in range(4)]))
    print("Average reward: %f" % average_reward)
    for p in range(4):
        plt.plot([max([basis.value(basis.feature(State(p, n), a), weight) for a in range(2)]) for n in range(11)], label="priority %d"%(1<<p))
    plt.grid(True)
    plt.legend()
    plt.show()
