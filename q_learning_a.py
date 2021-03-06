import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.racetrack import layout_left, layout_right, State, init_state, visualize


def init_policy(racetrack):
    policies = np.zeros(racetrack.shape + (5, 5, 9))
    for vx in range(policies.shape[2]):
        for vy in range(policies.shape[3]):
            actions = State((0, 0), (vx, vy)).actions
            p = 1 / len(actions)
            for a in actions:
                policies[:, :, vx, vy, a] = p
    return policies


def execute_policy(policy):
    return np.random.choice(range(len(policy)), p=policy)


def train(racetrack, episodes, alpha, epsilon):
    policies = init_policy(racetrack)
    values = np.zeros(racetrack.shape + (5, 5, 9))
    for _ in range(episodes):
        state = init_state(racetrack)
        while not state is None:
            action = execute_policy(policies[state.index])
            index = state.index + (action,)
            s1, r, _ = state.transition(racetrack, action)
            if s1 is None:
                values[index] += alpha * (r - values[index])
            else:
                values[index] += alpha * (r + max([values[s1.index][a] for a in s1.actions]) - values[index])
            state_actions = state.actions
            optimum = state_actions[np.argmax([values[state.index][a] for a in state_actions])]
            for a in state_actions:
                policies[state.index][a] = 1 - epsilon + epsilon / len(state_actions) if a == optimum else epsilon / len(state_actions)
            state = s1
    return np.argmax(policies, axis=-1), values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-learning: Off-policy TD(0) control (Section 6.5, Exercise 5.12).")
    parser.add_argument("--layout", help="racetrack layout of Figure 5.5 (default: left)", type=str, default="left")
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
    args = parser.parse_args()

    if args.layout.lower() == "left":
        racetrack = layout_left()
    elif args.layout.lower() == "right":
        racetrack = layout_right()
    else:
        raise ValueError("Invalid layout")

    print("Training...")
    policies, _ = train(racetrack, args.episodes, args.alpha, args.epsilon)
    print("Done!")

    visualize(racetrack, policies)
    plt.show()
