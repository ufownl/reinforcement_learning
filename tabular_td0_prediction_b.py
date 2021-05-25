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


def train(episodes, alpha, stick_threshold):
    values = np.zeros((2, 10, 10))
    for _ in range(episodes):
        s = init_state()
        while True:
            idx = s.index(False)
            s1, r = s.hit() if s.player_sum < stick_threshold else s.stick()
            if s1 is None:
                values[idx] += alpha * (r - values[idx])
                break
            else:
                values[idx] += alpha * (r + values[s1.index(False)] - values[idx])
            s = s1
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="first-visit MC prediction (Example 5.1).")
    parser.add_argument("--episodes", help="number of episodes (default: 500000)", type=int, default=500000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.01)", type=float, default=0.01)
    parser.add_argument("--stick_threshold", help="lower threshold of stick action (default: 20)", type=int, default=20)
    args = parser.parse_args()

    values = train(args.episodes, args.alpha, args.stick_threshold)
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
