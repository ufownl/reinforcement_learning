import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.blackjack import deal_card, State


def init_state():
    usable_ace = False
    player_sum = 0
    while player_sum < 12:
        card = deal_card()
        if player_sum < 11 and card == 1:
            usable_ace = True
            player_sum += 11
        else:
            player_sum += card
    return State(usable_ace, player_sum, deal_card())


def execute_policy(policy):
    x = random.random()
    ax = 0
    for i, p in enumerate(policy):
        if ax <= x and x < ax + p:
            return i
        ax += p
    return None


def train(episodes):
    values = np.zeros((2, 10, 10, 2))
    c = np.zeros((2, 10, 10, 2))
    behavior_policies = np.ones((2, 10, 10, 2)) * 0.5
    target_policies = np.argmax(values, axis=-1)
    for _ in range(episodes):
        s = init_state()
        episode = [(s, None, execute_policy(behavior_policies[s.index(False)]))]
        while True:
            s, _, a = episode[-1]
            s1, r1 = s.hit() if a == 0 else s.stick()
            if s1 is None:
                episode.append((s1, r1, None))
                break
            else:
                episode.append((s1, r1, execute_policy(behavior_policies[s1.index(False)])))
        g = 0
        w = 1
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            s, _, a = episode[t]
            idx = s.index(False) + (a,)
            c[idx] += w
            values[idx] += (g - values[idx]) * w / c[idx] if c[idx] > 0 else 0
            optimum = np.argmax(values[idx[:-1]])
            target_policies[idx[:-1]] = optimum
            if a != optimum:
                break
            w /= behavior_policies[idx]
    return target_policies, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Off-policy MC control (Section 5.7).")
    parser.add_argument("--episodes", help="number of episodes (default: 1000000)", type=int, default=1000000)
    args = parser.parse_args()

    policies, values = train(args.episodes)
    print(policies)

    print("Simulation Test:")
    wins = 0
    loses = 0
    draws = 0
    for _ in range(args.episodes):
        state = init_state()
        while not state is None:
            action = policies[state.index(False)]
            state, reward = state.hit() if action == 0 else state.stick()
        if reward > 0:
            wins += 1
        elif reward < 0:
            loses += 1
        else:
            draws += 1
    print("Wins %d / %d" % (wins, args.episodes))
    print("Loses %d / %d" % (loses, args.episodes))
    print("Draws %d / %d" % (draws, args.episodes))

    x, y = np.meshgrid(np.linspace(1, 10, 10), np.linspace(12, 21, 10))
    ax = plt.subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x, y, np.where(policies[0], values[0, :, :, 1], values[0, :, :, 0]))
    ax.set_xlabel("dealer showing card")
    ax.set_ylabel("player sum")
    ax.set_zlabel("no usable ace")
    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.plot_surface(x, y, np.where(policies[1], values[1, :, :, 1], values[1, :, :, 0]))
    ax.set_xlabel("dealer showing card")
    ax.set_ylabel("player sum")
    ax.set_zlabel("usable ace")
    plt.show()
