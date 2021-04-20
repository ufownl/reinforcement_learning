import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def deal_card():
    return random.choice(sum([[i] * 4 for i in range(1, 10)], start=[]) + [10] * 16)


class State:
    def __init__(self, usable_ace, player_sum, dealer_card):
        self.__usable_ace = 1 if usable_ace else 0
        if player_sum < 12:
            self.__player_sum = 0
        elif player_sum > 21:
            self.__player_sum = 9
        else:
            self.__player_sum = player_sum - 12
        if dealer_card < 1:
            self.__dealer_card = 0
        elif dealer_card > 10:
            self.__dealer_card = 9
        else:
            self.__dealer_card = dealer_card - 1

    @property
    def usable_ace(self):
        return self.__usable_ace != 0

    @property
    def player_sum(self):
        return self.__player_sum + 12

    @property
    def dealer_card(self):
        return self.__dealer_card + 1

    def index(self, flat=True):
        if flat:
            return self.__usable_ace * 100 + self.__player_sum * 10 + self.__dealer_card
        else:
            return (self.__usable_ace, self.__player_sum, self.__dealer_card)

    def hit(self):
        player_sum = self.player_sum + deal_card()
        if player_sum > 21:
            if self.usable_ace:
                return (State(False, player_sum - 10, self.dealer_card), 0)
            else:
                return (None, -1)
        else:
            return (State(self.usable_ace, player_sum, self.dealer_card), 0)

    def stick(self):
        dealer_sum = 11 if self.dealer_card == 1 else self.dealer_card
        while dealer_sum < 17:
            card = deal_card()
            dealer_sum += 11 if dealer_sum < 11 and card == 1 else card
        if dealer_sum > 21 or self.player_sum > dealer_sum:
            return (None, 1)
        elif self.player_sum == dealer_sum:
            return (None, 0)
        else:
            return (None, -1)


def init_episode():
    return [(State(random.choice([False, True]), random.randint(12, 21), random.randint(1, 10)), None, random.randint(0, 1))]


def train(episodes):
    policies = np.zeros((2, 10, 10), dtype=int)
    policies[:, -2:, :] = 1
    values = np.zeros((2, 10, 10, 2))
    n = np.zeros((2, 10, 10, 2), dtype=int)
    for _ in range(episodes):
        episode = init_episode()
        while True:
            s, _, a = episode[-1]
            s1, r1 = s.hit() if a == 0 else s.stick()
            if s1 is None:
                episode.append((s1, r1, None))
                break
            else:
                episode.append((s1, r1, policies[s1.index(False)]))
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            s, _, a = episode[t]
            # if all([(episode[i][0].index(), episode[i][2]) != (s.index(), a) for i in range(t)]):
            #     In this case, it is impossible for S[t] to occur earlier in an episode
            idx = s.index(False) + (a,)
            n[idx] += 1
            values[idx] += (g - values[idx]) / n[idx]
            policies[idx[:-1]] = np.argmax(values[idx[:-1]])
    return policies, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo ES (Example 5.3).")
    parser.add_argument("--episodes", help="number of episodes (default: 1000000)", type=int, default=1000000)
    args = parser.parse_args()

    policies, values = train(args.episodes)
    print(policies)
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