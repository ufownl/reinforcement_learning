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
            for i, a in enumerate(actions):
                policies[:, :, vx, vy, a] = p
    return policies


def execute_policy(policy):
    x = random.random()
    ax = 0
    for i, p in enumerate(policy):
        if ax <= x and x < ax + p:
            return i
        ax += p
    return None


def train(racetrack, episodes, alpha, epsilon, steps):
    policies = init_policy(racetrack)
    values = np.zeros(racetrack.shape + (5, 5, 9))
    for _ in range(episodes):
        s = init_state(racetrack)
        episode = [(s, None, execute_policy(policies[s.index]))]
        terminal_t = float("inf")
        t = 0
        while True:
            if t < terminal_t:
                s, _, a = episode[-1]
                s1, r1, _ = s.transition(racetrack, a)
                if s1 is None:
                    a1 = None
                    terminal_t = t + 1
                else:
                    a1 = execute_policy(policies[s1.index])
                episode.append((s1, r1, a1))
            update_t = t - steps + 1
            if update_t >= 0:
                g = sum([episode[i][1] for i in range(update_t + 1, min(update_t + steps, terminal_t) + 1)])
                if update_t + steps < terminal_t:
                    s, _, a = episode[update_t + steps]
                    g += values[s.index + (a,)]
                state, _, action = episode[update_t]
                index = state.index + (action,)
                values[index] += alpha * (g - values[index])
                state_actions = state.actions
                optimum = state_actions[np.argmax([values[state.index + (a,)] for a in state_actions])]
                for a in state_actions:
                    policies[state.index + (a,)] = 1 - epsilon + epsilon / len(state_actions) if a == optimum else epsilon / len(state_actions) 
            if update_t == terminal_t - 1:
                break
            t += 1
    return np.argmax(policies, axis=-1), values



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-step Sarsa (Section 7.2, Excercise 5.12).")
    parser.add_argument("--layout", help="racetrack layout of Figure 5.5 (default: left)", type=str, default="left")
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
    parser.add_argument("--steps", help="number of steps before bootstrapping", type=int, default=8)
    args = parser.parse_args()

    if args.layout.lower() == "left":
        racetrack = layout_left()
    elif args.layout.lower() == "right":
        racetrack = layout_right()
    else:
        raise ValueError("Invalid layout")

    print("Training...")
    policies, _ = train(racetrack, args.episodes, args.alpha, args.epsilon, args.steps)
    print("Done!")

    visualize(racetrack, policies)
    plt.show()