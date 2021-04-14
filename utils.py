import numpy as np


class Runner:
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def __call__(self, x):
        return np.expand_dims(x.run(*self.__args, **self.__kwargs), 0)
