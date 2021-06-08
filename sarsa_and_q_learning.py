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


def sarsa(world, episodes, alpha, epsilon):
    values = np.zeros(world.shape + (4,))
    for _ in range(episodes):
        state = State(np.array(np.where(world == 1)).transpose().tolist()[0])
        action = execute_policy(values[state.index], epsilon)
        while True:
            index = state.index + (action,)
            s1, r = state.transition(world, action)
            if s1 is None:
                values[index] += alpha * (r - values[index])
                break
            a1 = execute_policy(values[s1.index], epsilon)
            values[index] += alpha * (r + values[s1.index][a1] - values[index])
            state = s1
            action = a1
    return np.argmax(values, axis=-1), values


def q_learning(world, episodes, alpha, epsilon):
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
            values[index] += alpha * (r + np.max(values[s1.index]) - values[index])
            state = s1
    return np.argmax(values, axis=-1), values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Sarsa and Q-learning (Example 6.6).")
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()

    world = init_gridworld()
    print("Sarsa training...")
    sarsa_policies, _ = sarsa(world, args.episodes, args.alpha, args.epsilon)
    print("Q-learning training...")
    q_learning_policies, _ = q_learning(world, args.episodes, args.alpha, args.epsilon)
    print("Done!")

    plt.subplot(2, 1, 1)
    visualize(world, sarsa_policies, "blue")
    plt.subplot(2, 1, 2)
    visualize(world, q_learning_policies, "red")
    plt.show()
