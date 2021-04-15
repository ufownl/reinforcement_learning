import argparse
import numpy as np


class State:
    def __init__(self, pos, size):
        self.position = pos
        if pos == (0, 0) or pos == tuple([x - 1 for x in size]):
            self.actions = []
        else:
            directions = [
                (-1, 0) if pos[0] > 0 else (0, 0),
                (1, 0) if pos[0] < size[0] - 1 else (0, 0),
                (0, -1) if pos[1] > 0 else (0, 0),
                (0, 1) if pos[1] < size[1] - 1 else (0, 0)
            ]
            self.actions = [(pos[0] + x, pos[1] + y) for x, y in directions]


class GridWorld:
    def __init__(self, size):
        self.values = np.zeros(size)
        self.__states = [State((i, j), size) for i in range(size[0]) for j in range(size[1])]

    def train(self, theta, max_iters):
        for i in range(max_iters):
            delta = 0
            for s in self.__states:
                if s.actions:
                    v = self.values[s.position]
                    self.values[s.position] = -1 + 0.25 * sum([self.values[a] for a in s.actions])
                    delta = max(delta, abs(v - self.values[s.position]))
            if delta < theta:
                return i + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative Policy Evaluation (Example 4.1).")
    parser.add_argument("--size", help="size of the grid world (default: 4)", type=int, default=4)
    parser.add_argument("--theta", help="accuracy of estimation (default: 1e-10)", type=float, default=1e-10)
    parser.add_argument("--iters", help="max number of iterations (default: 10000)", type=int, default=10000)
    args = parser.parse_args()

    g = GridWorld((args.size, args.size))
    print(g.train(args.theta, args.iters))
    print(g.values)
