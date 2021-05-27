import numpy as np


class GridWorld:
    def __init__(self, size, starting_point, finish_point):
        self.__height, self.__width = size
        self.__starting_x, self.__starting_y = starting_point
        if self.__starting_x < 0 or self.__starting_x >= self.__height or self.__starting_y < 0 or self.__starting_y >= self.__width:
            raise ValueError("Invalid starting point")
        self.__finish_x, self.__finish_y = finish_point
        if self.__finish_x < 0 or self.__finish_x >= self.__height or self.__finish_y < 0 or self.__finish_y >= self.__width:
            raise ValueError("Invalid finish point")

    @property
    def size(self):
        return (self.__height, self.__width)

    @property
    def starting_point(self):
        return (self.__starting_x, self.__starting_y)

    @property
    def finish_point(self):
        return (self.__finish_x, self.__finish_y)


class State:
    __actions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (0, 0)
    ]

    def __init__(self, position):
        self.__pos_x, self.__pos_y = position

    @property
    def index(self):
        return (self.__pos_x, self.__pos_y)

    def transition(self, world, wind, action):
        h, w = world.size
        ax, ay = self.__actions[action]
        pos = (min(max(0, self.__pos_x + ax - wind[self.__pos_y]), h - 1), min(max(0, self.__pos_y + ay), w - 1))
        if pos == world.finish_point:
            return (None, 0)
        else:
            return (State(pos), -1)
