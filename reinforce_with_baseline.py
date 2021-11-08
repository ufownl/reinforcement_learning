import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.short_corridor import State, feature, policy, eligibility, value, dvalue


def generate_episode(theta):
    p = policy(theta)
    episode = [(State(), None, random.choices(list(range(2)), p, k=1)[0])]
    while True:
        s, _, a = episode[-1]
        s1, r = s.transition(a)
        if s1 is None:
            episode.append((s1, r, None))
            break
        else:
            episode.append((s1, r, random.choices(list(range(2)), p, k=1)[0]))
    return episode


def reinforce(episodes, alpha):
    theta = np.array([[-1.5], [1.5]])
    for _ in range(episodes):
        episode = generate_episode(theta)
        yield sum(r for _, r, _ in episode[1:])
        for t in range(0, len(episode) - 1):
            g = sum(r for _, r, _ in episode[t+1:])
            s, _, a = episode[t]
            theta = np.clip(theta + alpha * g * eligibility(theta)[:, a:a+1], -2.0, 2.0)


def reinforce_with_baseline(episodes, alpha_weight, alpha_theta):
    weight = np.array([0])
    theta = np.array([[-1.5], [1.5]])
    for _ in range(episodes):
        episode = generate_episode(theta)
        yield sum(r for _, r, _ in episode[1:])
        for t in range(0, len(episode) - 1):
            g = sum(r for _, r, _ in episode[t+1:])
            s, _, a = episode[t]
            delta = g - value(weight)
            weight = weight + alpha_weight * delta * dvalue(weight)
            theta = np.clip(theta + alpha_theta * delta * eligibility(theta)[:, a:a+1], -2.0, 2.0)
    

def episode_reward(f, *args, **kwargs):
    return np.array([reward for reward in f(*args, **kwargs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE with Baseline (Section 13.4, Example 13.1).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 100)", type=int, default=100)
    parser.add_argument("--episodes", help="number of episodes (default: 200)", type=int, default=200)
    args = parser.parse_args()

    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(reinforce, args.episodes, 2 ** -10), [episode_reward for _ in range(args.rounds)])), axis=0), label="REINFOCE")
        plt.plot(np.mean(np.concatenate(p.map(Runner(reinforce_with_baseline, args.episodes, 2 ** -10, 2 ** -6), [episode_reward for _ in range(args.rounds)])), axis=0), label="REINFOCE with Baseline")
    plt.grid(True)
    plt.legend()
    plt.show()
