import numpy as np


def init_dyna_maze():
    maze = np.zeros((6, 9), dtype=int)
    maze[2, 0] = 1
    maze[0, 8] = 2
    maze[1:4, 2] = -1
    maze[4, 5] = -1
    maze[0:3, 7] = -1
    return maze


class State:
    __actions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    def __init__(self, position):
        self.__pos_x, self.__pos_y = position

    @property
    def index(self):
        return (self.__pos_x, self.__pos_y)

    def transition(self, maze, action):
        h, w = maze.shape
        ax, ay = self.__actions[action]
        pos = (min(max(0, self.__pos_x + ax), h - 1), min(max(0, self.__pos_y + ay), w - 1))
        if maze[pos] < 0:
            return (State((self.__pos_x, self.__pos_y)), 0)
        elif maze[pos] == 2:
            return (None, 1)
        else:
            return (State(pos), 0)
