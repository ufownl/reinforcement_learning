import numpy as np


class Runner:
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, f):
        return np.expand_dims(f(*self.__args, **self.__kwargs), 0)


class StateFactory:
    def __init__(self, state, *args, **kwargs):
        self.__state = state
        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, x):
        return self.__state(x, *self.__args, **self.__kwargs)
