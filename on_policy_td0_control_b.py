import random
import argparse
import numpy as np
from utils.windy_gridworld import GridWorld, State


def execute_policy(policy):
    x = random.random()
    ax = 0
    for i, p in enumerate(policy):
        if ax <= x and x < ax + p:
            return i
        ax += p
    return None


def train(world, wind, episodes, alpha, epsilon, action_limit, stochastic_wind):
    policies = np.ones(world.size + (action_limit,)) / action_limit
    values = np.zeros_like(policies)
    for _ in range(episodes):
        state = State(world.starting_point)
        action = execute_policy(policies[state.index])
        while not state is None:
            index = state.index + (action,)
            s1, r = state.transition(world, [w + random.randint(-1, 1) if w > 0 else 0 for w in wind] if stochastic_wind else wind, action)
            if s1 is None:
                a1 = None
                values[index] += alpha * (r - values[index])
            else:
                a1 = execute_policy(policies[s1.index])
                values[index] += alpha * (r + values[s1.index + (a1,)] - values[index])
            optimum = np.argmax(values[state.index])
            for a in range(action_limit):
                policies[state.index + (a,)] = 1 - epsilon + epsilon / action_limit if a == optimum else epsilon / action_limit
            state = s1
            action = a1
    return policies, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarsa: on-policy TD(0) control (Example 6.5, Exercise 6.9, Exercise 6.10).")
    parser.add_argument("--episodes", help="number of training episodes (default: 10000)", type=int, default=10000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.5)", type=float, default=0.5)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--action_limit", help="number of preset actions that can be used (default: 4)", type=int, default=4)
    parser.add_argument("--stochastic_wind", help="using stochastic wind effect", action="store_true")
    args = parser.parse_args()

    world = GridWorld((7, 10), (3, 0), (3, 7))
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    print("Training...")
    policies, _ = train(world, wind, args.episodes, args.alpha, args.epsilon, args.action_limit, args.stochastic_wind)
    print("Done!")

    result = np.zeros(world.size, dtype=int)
    step = 0
    state = State(world.starting_point)
    while not state is None:
        step += 1
        result[state.index] = step
        action = np.argmax(policies[state.index])
        state, _ = state.transition(world, [w + random.randint(-1, 1) if w > 0 else 0 for w in wind] if args.stochastic_wind else wind, action)
    result[world.finish_point] = step + 1
    print(result)
    print("Total steps:", step + 1)
