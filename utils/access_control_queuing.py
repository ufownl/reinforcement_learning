import random
import numpy as np


class State:
    def __init__(self, priority=None, free_servers=10):
        if priority is None:
            self.__p = random.randrange(4)
        else:
            self.__p = priority
        self.__n = free_servers

    @property
    def priority(self):
        return self.__p

    @property
    def free_servers(self):
        return self.__n

    def transition(self, action):
        n = sum([1 for _ in range(10 - self.__n) if random.random() < 0.06])
        if self.__n > 0:
            return State(free_servers=self.__n-action+n), 1 << self.__p if action > 0 else 0
        else:
            return State(free_servers=n), 0


class Tabular:
    def __init__(self, alpha, beta):
        self.__alpha = alpha
        self.__beta = beta

    @property
    def alpha(self):
        return self.__alpha

    @property
    def beta(self):
        return self.__beta

    @property
    def dimensions(self):
        return 4 * 11 * 2

    def feature(self, state, action):
        x = np.zeros((4, 11, 2))
        x[state.priority, state.free_servers, action] = 1
        return x.reshape((-1, 1))

    def value(self, x, weight):
        return np.matmul(np.transpose(weight), x).item()

    def policy(self, state, weight, epsilon=None):
        if not epsilon is None and epsilon > 0 and random.random() < epsilon:
            return random.randrange(2)
        else:
            q = [self.value(self.feature(state, a), weight) for a in range(2)]
            max_q = max(q)
            return random.choice([i for i, v in enumerate(q) if v == max_q])
