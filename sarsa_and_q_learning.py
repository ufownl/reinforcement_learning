import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def init_gridworld():
    world = np.zeros((4, 12), dtype=int)
    world[3, 1:-1] = -1
    world[3, 0] = 1
    world[3, -1] = 2
    return world


class State:
    __actions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    def __init__(self, position):
        self.__pos_x, self.__pos_y = position

    @property
    def index(self):
        return (self.__pos_x, self.__pos_y)

    def transition(self, world, action):
        h, w = world.shape
        ax, ay = self.__actions[action]
        pos = (min(max(0, self.__pos_x + ax), h - 1), min(max(0, self.__pos_y + ay), w - 1))
        if world[pos] < 0:
            return (State(np.array(np.where(world == 1)).transpose().tolist()[0]), -100)
        elif world[pos] == 2:
            return (None, 0)
        else:
            return (State(pos), -1)


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
            values[index] += alpha * (r + values[s1.index + (a1,)] - values[index])
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


def visualize_path(policies, color):
    result = init_gridworld()
    state = State(np.array(np.where(world == 1)).transpose().tolist()[0])
    while True:
        action = policies[state.index]
        state, _ = state.transition(world, action)
        if state is None:
            break
        result[state.index] = 3
    plt.imshow(result + 1, cmap=ListedColormap(["gray", "white", "darkorange", "green", color]), vmax=4)


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
    visualize_path(sarsa_policies, "blue")
    plt.subplot(2, 1, 2)
    visualize_path(q_learning_policies, "red")
    plt.show()
