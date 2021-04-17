import math
import argparse
import numpy as np
from multiprocessing import cpu_count, Pool
from utils import StateFactory


def poisson(n, e):
    return e ** n / math.factorial(n) * math.exp(-e)


class State:
    def __init__(self, cars, req_e, ret_e, rental, mv_cost, max_cars, max_mv_cars):
        self.cars = cars
        self.updates = {}
        for i in range(max_cars + 1):
            rent_i = min(i, cars[0])
            prob_i = poisson(i, req_e[0])
            for j in range(max_cars + 1):
                rent_j = min(j, cars[1])
                prob_j = poisson(j, req_e[1])
                m_cars = (cars[0] - rent_i, cars[1] - rent_j)
                reward = rental * (rent_i + rent_j)
                for x in range(max_cars + 1):
                    cars_x = min(m_cars[0] + x, max_cars)
                    prob_x = poisson(x, ret_e[0])
                    for y in range(max_cars + 1):
                        cars_y = min(m_cars[1] + y, max_cars)
                        prob_y = poisson(y, ret_e[1])
                        key = (cars_x * (max_cars + 1) + cars_y, reward)
                        if key in self.updates:
                            self.updates[key] += prob_i * prob_j * prob_x * prob_y
                        else:
                            self.updates[key] = prob_i * prob_j * prob_x * prob_y
        self.actions = {}
        for m in range(-max_mv_cars, max_mv_cars + 1):
            if cars[0] - m >= 0 and cars[0] - m <= max_cars and cars[1] + m >= 0 and cars[1] + m <= max_cars:
                self.actions[m] = ((cars[0] - m) * (max_cars + 1) + cars[1] + m, -mv_cost * abs(m))


class CarRental:
    def __init__(self, req_e, ret_e, rental, mv_cost, max_cars, max_mv_cars):
        self.values = np.zeros((max_cars + 1, max_cars + 1))
        self.policies = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
        with Pool(cpu_count()) as p:
            self.__states = p.map(StateFactory(State, req_e, ret_e, rental, mv_cost, max_cars, max_mv_cars), [(i, j) for i in range(max_cars + 1) for j in range(max_cars + 1)])

    def train(self, gamma, theta, max_iters):
        for i in range(max_iters):
            while True:
                delta = 0
                for s in self.__states:
                    v = self.values[s.cars]
                    s1, r1 = s.actions[self.policies[s.cars]]
                    self.values[s.cars] = sum([self.__states[s1].updates[(s2, r2)] * (r1 + r2 + gamma * self.values[self.__states[s2].cars]) for s2, r2 in self.__states[s1].updates])
                    delta = max(delta, abs(v - self.values[s.cars]))
                if delta < theta:
                    break
            stable = True
            for s in self.__states:
                old = self.policies[s.cars]
                v_max = 0
                for a in s.actions:
                    s1, r1 = s.actions[a]
                    v = sum([self.__states[s1].updates[(s2, r2)] * (r1 + r2 + gamma * self.values[self.__states[s2].cars]) for s2, r2 in self.__states[s1].updates])
                    if v > v_max:
                        v_max = v
                        self.policies[s.cars] = a
                if old != self.policies[s.cars]:
                    stable = False
            if stable:
                break
            print("Policy", i + 1)
            print(self.policies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Iteration (Example 4.2).")
    parser.add_argument("--req_e1", help="expected number of customers at the first location (default: 3)", type=int, default=3)
    parser.add_argument("--req_e2", help="expected number of customers at the second location (default: 4)", type=int, default=4)
    parser.add_argument("--ret_e1", help="expected number of return cars at the first location (default: 3)", type=int, default=3)
    parser.add_argument("--ret_e2", help="expected number of return cars at the second location (default: 2)", type=int, default=2)
    parser.add_argument("--rental", help="rental per car (default: 10)", type=int, default=10)
    parser.add_argument("--mv_cost", help="moving cost per car (default: 2)", type=int, default=2)
    parser.add_argument("--max_cars", help="maximum number of cars per location (default: 20)", type=int, default=20)
    parser.add_argument("--max_mv_cars", help="maximum number of cars that can be moved per night (default: 5)", type=int, default=5)
    parser.add_argument("--gamma", help="discount rate of the future rewards (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--theta", help="accuracy of estimation (default: 1e-10)", type=float, default=1e-10)
    parser.add_argument("--iters", help="max number of iterations (default: 10000)", type=int, default=10000)
    args = parser.parse_args()

    print("Initializing...")
    g = CarRental((args.req_e1, args.req_e2), (args.ret_e1, args.ret_e2), args.rental, args.mv_cost, args.max_cars, args.max_mv_cars)
    print("Training...")
    g.train(args.gamma, args.theta, args.iters)
    print("Value")
    print(g.values)
