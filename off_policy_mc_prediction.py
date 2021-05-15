import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from blackjack import State
from utils import Runner


def execute_policy(policy):
    x = random.random()
    ax = 0
    for i, p in enumerate(policy):
        if ax <= x and x < ax + p:
            return i
        ax += p
    return None


def on_policy_predict(episodes, state, policies):
    values = np.zeros((2, 10, 10))
    n = np.zeros((2, 10, 10), dtype=int)
    for _ in range(episodes):
        episode = [(state, None, execute_policy(policies[state.index(False)]))]
        while True:
            s, _, a = episode[-1]
            s1, r1 = s.hit() if a == 0 else s.stick()
            if s1 is None:
                episode.append((s1, r1, None))
                break
            else:
                episode.append((s1, r1, execute_policy(policies[s1.index(False)])))
        g = 0
        for t in reversed(range(len(episode) - 1)):
            g += episode[t + 1][1]
            s, _, _ = episode[t]
            # if all([episode[i][0].index() != s.index() for i in range(t)]):
            #     In this case, it is impossible for S[t] to occur earlier in an episode
            idx = s.index(False)
            n[idx] += 1
            values[idx] += (g - values[idx]) / n[idx]
    return values[state.index(False)]


def ordinary_importance_sampling(episodes, state, target_policies, behavior_policies):
    values = np.zeros((2, 10, 10, 2))
    n = np.zeros((2, 10, 10, 2), dtype=int)
    for _ in range(episodes):
        episode = [(state, None, execute_policy(behavior_policies[state.index(False)]))]
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
            n[idx] += 1
            values[idx] += (g - values[idx]) * w / n[idx]
            w *= target_policies[idx] / behavior_policies[idx]
        yield values[state.index(False) + (execute_policy(target_policies[state.index(False)]),)]


def weighted_importance_sampling(episodes, state, target_policies, behavior_policies):
    values = np.zeros((2, 10, 10, 2))
    c = np.zeros((2, 10, 10, 2))
    for _ in range(episodes):
        episode = [(state, None, execute_policy(behavior_policies[state.index(False)]))]
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
            w *= target_policies[idx] / behavior_policies[idx]
        yield values[state.index(False) + (execute_policy(target_policies[state.index(False)]),)]


class OffPolicyPrediction:
    def __init__(self, f):
        self.__f = f

    def run(self, episodes, state, target_policies, behavior_policies):
        return np.array([v for v in self.__f(episodes, state, target_policies, behavior_policies)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="off-policy MC prediction (Example 5.4, Section 5.6).")
    parser.add_argument("--dealer_card", help="dealer's showing card at the initial state (default: 2)", type=int, default=2)
    parser.add_argument("--player_sum", help="sum of the player's cards at the initial state (default: 13)", type=int, default=13)
    parser.add_argument("--no_usable_ace", help="player has no usable ace at the initial state", action="store_true")
    parser.add_argument("--stick_threshold", help="lower threshold of stick action of the target policy (default: 20)", type=int, default=20)
    parser.add_argument("--on_policy_episodes", help="number of episodes of on-policy prediction (default: 1000000)", type=int, default=1000000)
    parser.add_argument("--off_policy_episodes", help="number of episodes of off-policy prediction (default: 10000)", type=int, default=10000)
    parser.add_argument("--off_policy_rounds", help="number of running rounds of off-policy prediction (default: 100)", type=int, default=100)
    args = parser.parse_args()

    print("Generating initial state...")
    state = State(not args.no_usable_ace, args.player_sum, args.dealer_card)
    print("Generating target policy...")
    policies = np.zeros((2, 10, 10, 2))
    stick_threshold = args.stick_threshold - 22
    policies[:, :stick_threshold, :, 0] = 1.0
    policies[:, stick_threshold:, :, 1] = 1.0
    print("Evaluating expected target value...")
    target = on_policy_predict(args.on_policy_episodes, state, policies)
    print("Expected target value:", target)
    with Pool(cpu_count()) as p:
        ordinary_mse = np.mean((np.concatenate(p.map(Runner(args.off_policy_episodes, state, policies, np.ones_like(policies) * 0.5), [OffPolicyPrediction(ordinary_importance_sampling) for _ in range(args.off_policy_rounds)])) - target) ** 2, axis=0)
        weighted_mse = np.mean((np.concatenate(p.map(Runner(args.off_policy_episodes, state, policies, np.ones_like(policies) * 0.5), [OffPolicyPrediction(weighted_importance_sampling) for _ in range(args.off_policy_rounds)])) - target) ** 2, axis=0)
    print("MSE of ordinary importance sampling:", ordinary_mse)
    print("MSE of weighted importance sampling:", weighted_mse)

    plt.plot(ordinary_mse, scaley=False, label="Ordinary importance sampling")
    plt.plot(weighted_mse, scaley=False, label="Weighted importance sampling")
    plt.xscale("log")
    plt.ylim(ymin=-0.2, ymax=5)
    plt.grid(True)
    plt.legend()
    plt.show()
