import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.racetrack import layout_left, layout_right, State, init_state, visualize


def init_value(racetrack):
    values = np.ones(racetrack.shape + (5, 5, 9)) * float("-inf")
    for vx in range(values.shape[2]):
        for vy in range(values.shape[3]):
            for a in State((0, 0), (vx, vy)).actions:
                values[:, :, vx, vy, a] = 0
    return values


def greedy_policy(value):
    max_q = np.max(value)
    return random.choice([a for a, q in enumerate(value) if q == max_q])


def train(racetrack, episodes):
    values = init_value(racetrack)
    for _ in range(episodes):
        state = init_state(racetrack)
        while not state is None:
            action = greedy_policy(values[state.index])
            next_state, _, _ = state.transition(racetrack, action)
            values[state.index][action] = sum([p * (r + np.max(values[s1.index])) for s1, r, p in state.branches(racetrack, action) if not s1 is None])
            state = next_state
    return np.argmax(values, axis=-1), values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Dynamic Programming (Section 8.7, Exercise 5.12).")
    parser.add_argument("--layout", help="racetrack layout of Figure 5.5 (default: left)", type=str, default="left")
    parser.add_argument("--episodes", help="number of training episodes (default: 10000)", type=int, default=10000)
    args = parser.parse_args()

    if args.layout.lower() == "left":
        racetrack = layout_left()
    elif args.layout.lower() == "right":
        racetrack = layout_right()
    else:
        raise ValueError("Invalid layout")

    print("Training...")
    policies, _ = train(racetrack, args.episodes)
    print("Done!")

    visualize(racetrack, policies)
    plt.show()
