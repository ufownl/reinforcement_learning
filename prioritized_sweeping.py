import math
import heapq
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from utils.parallel import Runner
from utils.maze import init_dyna_maze, State


def execute_policy(value, epsilon):
    if np.max(value) == np.min(value) or random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def pqueue_push(pqueue, priority, entry):
    for i, (p, e) in enumerate(pqueue):
        if e == entry:
            if -priority < p:
                pqueue[i] = (-priority, entry)
                heapq.heapify(pqueue)
            return
    heapq.heappush(pqueue, (-priority, entry))


def prioritized_sweeping(maze, time_steps, alpha, gamma, epsilon, theta, planning_steps):
    values = np.zeros(maze.shape + (4,))
    model = {}
    predecessors = {}
    pqueue = []
    t = 0
    backups = 0
    yield backups
    while True:
        state = State(np.array(np.where(maze == 1)).transpose().tolist()[0])
        while not state is None:
            action = execute_policy(values[state.index], epsilon)
            s1, r = state.transition(maze, action)
            if state.index in model:
                model[state.index][action] = (s1, r)
            else:
                model[state.index] = {action: (s1, r)}
            if not s1 is None:
                if s1.index in predecessors:
                    predecessors[s1.index][(state.index, action)] = r
                else:
                    predecessors[s1.index] = {(state.index, action): r}
            priority = abs(r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[state.index][action])
            if priority > theta:
                pqueue_push(pqueue, priority, (state.index, action))
            state = s1
            for _ in range(planning_steps):
                if pqueue == []:
                    break
                _, (s, a) = heapq.heappop(pqueue)
                s1, r = model[s][a]
                values[s][a] += alpha * (r + (0 if s1 is None else gamma * np.max(values[s1.index])) - values[s][a])
                backups += 1
                if s in predecessors:
                    for s0, a0 in predecessors[s]:
                        r0 = predecessors[s][(s0, a0)]
                        priority = abs(r0 + gamma * np.max(values[s]) - values[s0][a0])
                        if priority > theta:
                            pqueue_push(pqueue, priority, (s0, a0))
            yield backups
            t += 1
            if t >= time_steps:
                return


class CumulativeBackups:
    def __init__(self, f):
        self.__f = f

    def __call__(self, *args, **kwargs):
        return np.array([v for v in self.__f(*args, **kwargs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prioritized sweeping for a deterministic environment (Section 8.4).")
    parser.add_argument("--rounds", help="total number of running rounds (default: 30)", type=int, default=30)
    parser.add_argument("--alpha", help="constant step-size parameter of TD (default: 1.0)", type=float, default=1.0)
    parser.add_argument("--gamma", help="discount rate of reward (default: 0.95)", type=float, default=0.95)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--theta", help="threshold of value effect (default: 1e-4)", type=float, default=1e-4)
    parser.add_argument("--planning_steps", help="number of planning steps (default: 5)", type=int, default=5)
    args = parser.parse_args()

    maze = init_dyna_maze()
    plt.plot([args.planning_steps * t for t in range(1500)], label="Dyna-Q")
    print("Training...")
    with Pool(cpu_count()) as p:
        plt.plot(np.mean(np.concatenate(p.map(Runner(maze, 1500, args.alpha, args.gamma, args.epsilon, args.theta, args.planning_steps), [CumulativeBackups(prioritized_sweeping) for _ in range(args.rounds)])), 0), label="Prioritized sweeping")
    print("Done!")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()
