import random
import numpy as np


class State:
    def __init__(self, index=499):
        if index < 0 or index > 999:
            raise IndexError("Invalid state")
        self.__index = index

    @property
    def index(self):
        return self.__index

    def transition(self):
        index = random.choice(list(range(self.__index - 100, self.__index)) + list(range(self.__index + 1, self.__index + 101)))
        if index < 0:
            return (None, -1)
        elif index > 999:
            return (None, 1)
        else:
            return (State(index), 0)


def calc_state_values(theta):
    values = np.zeros(1000)
    while True:
        delta = 0
        for s in range(len(values)):
            v = values[s]
            values[s] = (sum([-1 if s1 < 0 else values[s1] for s1 in range(s - 100, s)]) + sum([1 if s1 > 999 else values[s1] for s1 in range(s + 1, s + 101)])) / 200
            delta = max(delta, abs(v - values[s]))
        if delta < theta:
            break
    return values
