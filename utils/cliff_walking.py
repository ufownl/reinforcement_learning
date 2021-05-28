import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def init_gridworld():
    world = np.zeros((4, 12), dtype=int)
    world[3, 1:-1] = -1
    world[3, 0] = 1
    world[3, -1] = 2
    return world


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

    def transition(self, world, action):
        h, w = world.shape
        ax, ay = self.__actions[action]
        pos = (min(max(0, self.__pos_x + ax), h - 1), min(max(0, self.__pos_y + ay), w - 1))
        if world[pos] < 0:
            return (State(np.array(np.where(world == 1)).transpose().tolist()[0]), -100)
        elif world[pos] == 2:
            return (None, 0)
        else:
            return (State(pos), -1)


def visualize(world, policies, color):
    result = np.copy(world)
    state = State(np.array(np.where(world == 1)).transpose().tolist()[0])
    while True:
        action = policies[state.index]
        state, _ = state.transition(world, action)
        if state is None:
            break
        result[state.index] = 3
    plt.imshow(result + 1, cmap=ListedColormap(["gray", "white", "darkorange", "green", color]), vmax=4)
