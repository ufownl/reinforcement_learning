import random
import argparse
import numpy as np
from utils.windy_gridworld import GridWorld, State


def execute_policy(value, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(value))
    else:
        return np.argmax(value)


def train(world, wind, episodes, alpha, epsilon, action_limit, stochastic_wind):
    values = np.zeros(world.size + (action_limit,))
    for _ in range(episodes):
        state = State(world.starting_point)
        action = execute_policy(values[state.index], epsilon)
        while True:
            index = state.index + (action,)
            s1, r = state.transition(world, [w + random.randint(-1, 1) if w > 0 else 0 for w in wind] if stochastic_wind else wind, action)
            if s1 is None:
                values[index] += alpha * (r - values[index])
                break
            a1 = execute_policy(values[s1.index], epsilon)
            values[index] += alpha * (r + values[s1.index][a1] - values[index])
            state = s1
            action = a1
    return np.argmax(values, axis=-1), values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sarsa: on-policy TD(0) control (Example 6.5, Exercise 6.9, Exercise 6.10).")
    parser.add_argument("--episodes", help="number of training episodes (default: 100000)", type=int, default=100000)
    parser.add_argument("--alpha", help="constant step-size parameter of TD(0) (default: 0.1)", type=float, default=0.1)
    parser.add_argument("--epsilon", help="probability of exploration (default: 0.2)", type=float, default=0.2)
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
        action = policies[state.index]
        state, _ = state.transition(world, [w + random.randint(-1, 1) if w > 0 else 0 for w in wind] if args.stochastic_wind else wind, action)
    result[world.finish_point] = step + 1
    print(result)
    print("Total steps:", step + 1)
