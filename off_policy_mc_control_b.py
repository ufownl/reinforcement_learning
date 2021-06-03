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
    x = random.random()
    ax = 0
    for i, p in enumerate(policy):
        if ax <= x and x < ax + p:
            return i
        ax += p
    return None


def generate_episode(racetrack, policies):
    s = init_state(racetrack)
    episode = [(s, None, execute_policy(policies[s.index]))]
    while True:
        s, _, a = episode[-1]
        s1, r1, _ = s.transition(racetrack, a)
        if s1 is None:
            episode.append((s1, r1, None))
            break
        else:
            episode.append((s1, r1, execute_policy(policies[s1.index])))
    return episode


def pretrain(racetrack, episodes, epsilon):
    policies = init_policy(racetrack)
    values = np.zeros(racetrack.shape + (5, 5, 9))
    n = np.zeros_like(values, dtype=int)
    for _ in range(episodes):
        episode = generate_episode(racetrack, policies)
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            state, _, action = episode[t]
            index = state.index + (action,)
            if all([episode[i][0].index + (episode[i][2],) != index for i in range(t)]):
                n[index] += 1
                values[index] += (g - values[index]) / n[index]
                state_actions = state.actions
                optimum = state_actions[np.argmax([values[state.index + (a,)] for a in state_actions])]
                for a in state_actions:
                    policies[state.index + (a,)] = 1 - epsilon + epsilon / len(state_actions) if a == optimum else epsilon / len(state_actions) 
    return policies, values


def train(racetrack, episodes, epsilon, behavior_policies, values):
    c = np.zeros_like(values)
    target_policies = np.argmax(behavior_policies, axis=-1)
    for _ in range(episodes):
        episode = generate_episode(racetrack, behavior_policies)
        g = 0
        w = 1
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            state, _, action = episode[t]
            index = state.index + (action,)
            c[index] += w
            values[index] += (g - values[index]) * w / c[index] if c[index] > 0 else 0
            state_actions = state.actions
            optimum = state_actions[np.argmax([values[state.index + (a,)] for a in state_actions])]
            for a in state_actions:
                behavior_policies[state.index + (a,)] = 1 - epsilon + epsilon / len(state_actions) if a == optimum else epsilon / len(state_actions) 
            target_policies[state.index] = optimum
            if action != optimum:
                break
            w /= behavior_policies[index]
    return target_policies, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Off-policy MC control (Exercise 5.12).")
    parser.add_argument("--layout", help="racetrack layout of Figure 5.5 (default: left)", type=str, default="left")
    parser.add_argument("--pretraining_episodes", help="number of pre-training episodes (default: 10000)", type=int, default=10000)
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
    args = parser.parse_args()

    if args.layout.lower() == "left":
        racetrack = layout_left()
    elif args.layout.lower() == "right":
        racetrack = layout_right()
    else:
        raise ValueError("Invalid layout")
    print("On-policy pre-training...")
    policies, values = pretrain(racetrack, args.pretraining_episodes, args.epsilon)
    print("Off-policy training...")
    policies, _ = train(racetrack, args.episodes, args.epsilon, policies, values)
    print("Done!")

    visualize(racetrack, policies)
    plt.show()
