import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.maze import init_dyna_maze, State


def execute_policy(value, epsilon):
    if np.max(value) == np.min(value) or random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def train(maze, episodes, alpha, gamma, epsilon, steps):
    values = np.zeros(maze.shape + (4,))
    model = {}
    for _ in range(episodes):
        state = State(np.array(np.where(maze == 1)).transpose().tolist()[0])
        t = 0
        while not state is None:
            action = execute_policy(values[state.index], epsilon)
            index = state.index + (action,)
            s1, r = state.transition(maze, action)
            values[index] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[index])
            if state.index in model:
                model[state.index][action] = (s1, r)
            else:
                model[state.index] = {action: (s1, r)}
            state = s1
            for _ in range(steps):
                s = random.choice(list(model))
                actions = model[s]
                a = random.choice(list(actions))
                s1, r = actions[a]
                values[s + (a,)] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[s + (a,)])
            t += 1
        yield t


def steps_per_episode(maze, episodes, alpha, gamma, epsilon, steps):
    return np.array([v for v in train(maze, episodes, alpha, gamma, epsilon, steps)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular Dyna-Q (Example 8.1).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 30)", type=int, default=30)
    parser.add_argument("--episodes", help="number of training episodes (default: 50)", type=int, default=50)
    parser.add_argument("--alpha", help="constant step-size parameter of TD (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--gamma", help="discount rate of reward (default: 0.95)", type=float, default=0.95)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    args = parser.parse_args()

    maze = init_dyna_maze()
    print("Training...")
    with Pool(cpu_count()) as p:
        for steps in [0, 5, 50]:
            plt.plot([e + 1 for e in range(args.episodes)], np.mean(np.concatenate(p.map(Runner(maze, args.episodes, args.alpha, args.gamma, args.epsilon, steps), [steps_per_episode for _ in range(args.rounds)])), 0), label="%d planning steps"%steps)
    print("Done!")
    plt.grid(True)
    plt.legend()
    plt.show()
