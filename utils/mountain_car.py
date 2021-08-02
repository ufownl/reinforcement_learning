import math
import random
import numpy as np


class State:
    def __init__(self, position=None, velocity=0.0):
        if position is None:
            self.__pos = random.uniform(-0.6, -0.4)
        else:
            self.__pos = position
        self.__vel = velocity

    @property
    def position(self):
        return self.__pos

    @property
    def velocity(self):
        return self.__vel

    def transition(self, action):
        v = self.__vel + 0.001 * action - 0.0025 * math.cos(3 * self.__pos)
        v = min(max(-0.07, v), 0.07)
        p = self.__pos + v
        if p >= 0.5:
            return None, 0
        elif p <= -1.2:
            return State(-1.2), -1
        else:
            return State(p, v), -1


class TileCoding:
    def __init__(self, alpha):
        self.__alpha = alpha / 8
        p = np.array([
            np.arange(9, dtype=int),
            np.arange(1, 10, dtype=int)
        ]) * 1.7 / 8 - 1.2
        self.__p = np.array([p - i * 1.7 / 64 for i in range(8)])
        v = np.array([
            np.arange(9, dtype=int),
            np.arange(1, 10, dtype=int)
        ]) * 0.14 / 8 - 0.07
        self.__v = np.array([v - i * 0.14 / 64 for i in range(8)])
        a = np.array([
            np.arange(9, dtype=int),
            np.arange(1, 10, dtype=int)
        ]) * 2.0 / 8 - 1.0
        self.__a = np.array([a - i * 2.0 / 64 for i in range(8)])

    @property
    def alpha(self):
        return self.__alpha

    @property
    def dimensions(self):
        return self.__p.shape[-1] * self.__v.shape[-1] * self.__a.shape[-1] * 8

    def feature(self, state, action):
        if state is None:
            return np.zeros((self.dimensions, 1))
        else:
            f = lambda x, a: np.argmax(np.logical_and(x >= a[0], x < a[1]))
            indics = [(f(state.position, self.__p[i]), f(state.velocity, self.__v[i]), f(action, self.__a[i])) for i in range(8)]
            features = [np.zeros((self.__p.shape[-1], self.__v.shape[-1], self.__a.shape[-1])) for i in range(8)]
            for i, (p, v, a) in enumerate(indics):
                features[i][p, v, a] = 1.0
                features[i] = features[i].reshape((-1,))
            return np.concatenate(features).reshape((-1, 1))

    def value(self, x, weight):
        return np.matmul(np.transpose(weight), x).item()

    def policy(self, state, weight, epsilon=None):
        actions = [-1.0, 0.0, 1.0]
        if not epsilon is None and epsilon > 0 and random.random() < epsilon:
            return random.choice(actions)
        else:
            q = [self.value(self.feature(state, a), weight) for a in actions]
            max_q = max(q)
            return random.choice([actions[i] for i, v in enumerate(q) if v == max_q])
