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


def train(racetrack, episodes, alpha, epsilon, steps):
    behavior_policies = init_policy(racetrack)
    values = np.zeros(racetrack.shape + (5, 5, 9))
    target_policies = np.argmax(behavior_policies, axis=-1)
    for _ in range(episodes):
        state = init_state(racetrack)
        episode = [(state, None, execute_policy(behavior_policies[state.index]))]
        t = 0
        while True:
            state, _, action = episode[-1]
            if not state is None:
                s1, r1, _ = state.transition(racetrack, action)
                episode.append((s1, r1, None if s1 is None else execute_policy(behavior_policies[s1.index])))
            update_t = t + 1 - steps
            if update_t >= 0:
                state, reward, _ = episode[-1]
                g = reward
                if not state is None:
                    g += values[state.index + (target_policies[state.index],)]
                for sk, rk, ak in reversed(episode[update_t+1:-1]):
                    a = target_policies[sk.index]
                    if a == ak:
                        g += rk
                    else:
                        g = rk + values[sk.index + (a,)]
                state, _, action = episode[update_t]
                index = state.index + (action,)
                values[index] += alpha * (g - values[index])
                state_actions = state.actions
                optimum = state_actions[np.argmax([values[state.index + (a,)] for a in state_actions])]
                for a in state_actions:
                    behavior_policies[state.index + (a,)] = 1 - epsilon + epsilon / len(state_actions) if a == optimum else epsilon / len(state_actions) 
                target_policies[state.index] = optimum
                if episode[update_t + 1][0] is None:
                    break
            t += 1
    return target_policies, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-step Tree Backup (Section 7.5, Excercise 5.12).")
    parser.add_argument("--layout", help="racetrack layout of Figure 5.5 (default: left)", type=str, default="left")
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
    parser.add_argument("--steps", help="number of steps before bootstrapping (default: 8)", type=int, default=8)
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
