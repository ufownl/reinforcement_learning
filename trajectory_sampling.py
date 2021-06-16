import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner


def generate_task(states, branches):
    return [[list(zip(random.sample(range(states - 1), k=branches), [random.gauss(0, 1) for _ in range(branches)], [0.9 / branches] * branches)) + [(states - 1, random.gauss(0, 1), 0.1)] for _ in range(2)] for _ in range(states - 1)]


class State:
    def __init__(self, index):
        self.__index = index

    @property
    def index(self):
        return self.__index

    def transition(self, task, action):
        branches = task[self.__index][action]
        s1, r, _ = branches[np.random.choice(range(len(branches)), p=list(zip(*branches))[-1])]
        if s1 >= len(task):
            return (None, r)
        else:
            return (State(s1), r)


def policy_evaluate(task, policies, episodes):
    g = 0
    for _ in range(episodes):
        s = State(0)
        while not s is None:
            s, r = s.transition(task, policies[s.index])
            g += r
    return g / episodes


def uniform_update(task, steps, eval_episodes):
    q = np.random.normal(size=(len(task) + 1, 2))
    q[-1, :] = 0
    t = 0
    yield 0
    while True:
        for s, actions in enumerate(task):
            for a, branches in enumerate(actions):
                q[s, a] = sum([p * (r + np.max(q[s1])) for s1, r, p in branches])
                t += 1
                if t % int(steps / 200) == 0:
                    yield policy_evaluate(task, np.argmax(q, axis=-1), eval_episodes)
                if t >= steps:
                    return


def on_policy_update(task, steps, epsilon, eval_episodes):
    q = np.random.normal(size=(len(task) + 1, 2))
    q[-1, :] = 0
    t = 0
    yield 0
    while True:
        s = State(0)
        while not s is None:
            if random.random() < epsilon:
                a = random.randrange(0, 2)
            else:
                a = np.argmax(q[s.index])
            next_s, _ = s.transition(task, a)
            q[s.index, a] = sum([p * (r + np.max(q[s1])) for s1, r, p in task[s.index][a]])
            s = next_s
            t += 1
            if t % int(steps / 200) == 0:
                yield policy_evaluate(task, np.argmax(q, axis=-1), eval_episodes)
            if t >= steps:
                return


class StartStateValue:
    def __init__(self, f):
        self.__f = f

    def __call__(self, *args, **kwargs):
        return np.array([v for v in self.__f(*args, **kwargs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Sampling (Exercise 8.8).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 20)", type=int, default=20)
    parser.add_argument("--states", help="number of task states (default: 10000)", type=int, default=10000)
    parser.add_argument("--branches", help="number of non-terminal branches per state-action pair (default: 3)", type=int, default=3)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--eval_episodes", help="number of Monte-Carlo evaluation episodes (default: 1000)", type=int, default=1000)
    args = parser.parse_args()

    task = generate_task(args.states, args.branches)
    steps = args.states * 20
    with Pool(cpu_count()) as p:
        print("Uniform updating...")
        plt.plot([i * (steps / 200) for i in range(201)], np.mean(np.concatenate(p.map(Runner(task, steps, args.eval_episodes), [StartStateValue(uniform_update) for _ in range(args.rounds)])), 0), label="uniform")
        print("On-policy updating...")
        plt.plot([i * (steps / 200) for i in range(201)], np.mean(np.concatenate(p.map(Runner(task, steps, args.epsilon, args.eval_episodes), [StartStateValue(on_policy_update) for _ in range(args.rounds)])), 0), label="on-policy")
    print("Done!")
    plt.grid(True)
    plt.legend()
    plt.show()
