import numpy as np


class State:
    def __init__(self, index=0):
        self.__index = index

    def transition(self, action):
        if self.__index == 1:
            index = self.__index + (-1 if action == 0 else 1)
        else:
            index = self.__index + (1 if action == 0 else -1)
        if index < 0:
            return State(0), -1
        elif index > 2:
            return None, -1
        else:
            return State(index), -1


def feature():
    return np.array([[1, 0], [0, 1]])


def policy(theta):
    h = np.matmul(np.transpose(theta), feature())
    e = np.exp(h[0])
    return e / np.sum(e)


def eligibility(theta):
    x = feature()
    p = policy(theta)
    return x - np.sum(p * x, axis=-1, keepdims=True)
