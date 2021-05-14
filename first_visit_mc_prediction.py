import argparse
import numpy as np
import matplotlib.pyplot as plt
from blackjack import deal_card, State


def init_episode():
    usable_ace = False
    player_sum = 0
    while player_sum < 12:
        card = deal_card()
        if player_sum < 11 and card == 1:
            usable_ace = True
            player_sum += 11
        else:
            player_sum += card
    return [(State(usable_ace, player_sum, deal_card()), None)]


def train(episodes, stick_threshold):
    values = np.zeros((2, 10, 10))
    n = np.zeros((2, 10, 10), dtype=int)
    for _ in range(episodes):
        episode = init_episode()
        while True:
            s, _ = episode[-1]
            if s is None:
                break
            episode.append(s.hit() if s.player_sum < stick_threshold else s.stick())
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            s, _ = episode[t]
            # if all([episode[i][0].index() != s.index() for i in range(t)]):
            #     In this case, it is impossible for S[t] to occur earlier in an episode
            idx = s.index(False)
            n[idx] += 1
            values[idx] += (g - values[idx]) / n[idx]
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="first-visit MC prediction (Example 5.1).")
    parser.add_argument("--episodes", help="number of episodes (default: 50000)", type=int, default=50000)
    parser.add_argument("--stick_threshold", help="lower threshold of stick action (default: 20)", type=int, default=20)
    args = parser.parse_args()

    values = train(args.episodes, args.stick_threshold)
    print(values)
    x, y = np.meshgrid(np.linspace(1, 10, 10), np.linspace(12, 21, 10))
    ax = plt.subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x, y, values[0])
    ax.set_xlabel("dealer showing card")
    ax.set_ylabel("player sum")
    ax.set_zlabel("no usable ace")
    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.plot_surface(x, y, values[1])
    ax.set_xlabel("dealer showing card")
    ax.set_ylabel("player sum")
    ax.set_zlabel("usable ace")
    plt.show()
