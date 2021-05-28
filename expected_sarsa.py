import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.cliff_walking import init_gridworld, State, visualize


def execute_policy(value, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def train(world, episodes, alpha, epsilon):
    values = np.zeros(world.shape + (4,))
    for _ in range(episodes):
        state = State(np.array(np.where(world == 1)).transpose().tolist()[0])
        while True:
            action = execute_policy(values[state.index], epsilon)
            index = state.index + (action,)
            s1, r = state.transition(world, action)
            if s1 is None:
                values[index] += alpha * (r - values[index])
                break
            p = np.ones(4) * epsilon / 4
            p[np.argmax(values[s1.index])] += 1 - epsilon
            values[index] += alpha * (r + np.sum(p * values[s1.index]) - values[index])
            state = s1
    return np.argmax(values, axis=-1), values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expected Sarsa (Section 6.6, Example 6.6).")
    parser.add_argument("--episodes", help="number of training episodes (default: 30)", type=int, default=30)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 1.0)", type=float, default=1.0)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()

    world = init_gridworld()
    print("Training...")
    policies, _ = train(world, args.episodes, args.alpha, args.epsilon)
    print("Done!")

    visualize(world, policies, "blue")
    plt.show()
