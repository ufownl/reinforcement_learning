import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def layout_left():
    racetrack = np.zeros((32, 17), dtype=np.int)
    # left
    racetrack[:3, :3] = -1
    racetrack[3:10, :2] = -1
    racetrack[10:18, 0] = -1
    racetrack[-4, 0] = -1
    racetrack[-3:-1, :2] = -1
    racetrack[-1, :3] = -1
    # right
    racetrack[:-7, -8:] = -1
    racetrack[-7, -7:] = -1
    # starting line
    racetrack[0, 3:9] = 1
    # finish line
    racetrack[-6:, -1] = 2
    return racetrack


def layout_right():
    racetrack = np.zeros((30, 32), dtype=np.int)
    # left
    for i in range(13):
        racetrack[3+i, :i+1] = -1
    racetrack[16:21, :14] = -1
    racetrack[21, :13] = -1
    racetrack[22, :12] = -1
    racetrack[23:27, :11] = -1
    racetrack[-3, :12] = -1
    racetrack[-2, :13] = -1
    racetrack[-1, :16] = -1
    # right
    racetrack[:17, -9:] = -1
    racetrack[17, -8:] = -1
    racetrack[18, -6:] = -1
    racetrack[19, -5:] = -1
    racetrack[20, -2:] = -1
    # starting line
    racetrack[0, :23] = 1
    # finish line
    racetrack[-9:, -1] = 2
    return racetrack


def visualize(layout):
    plt.imshow(layout + 1, cmap=ListedColormap(["gray", "white", "darkorange", "green", "red", "gold"]), vmax=5)


def bresenham_line(pt0, pt1):
    yield pt0
    dx = abs(pt1[0] - pt0[0])
    sx = 1 if pt0[0] < pt1[0] else -1
    dy = -abs(pt1[1] - pt0[1])
    sy = 1 if pt0[1] < pt1[1] else -1
    e = dx + dy
    while pt0 != pt1:
        e2 = e * 2
        if e2 >= dy:
            e += dy
            pt0 = (pt0[0] + sx, pt0[1])
        if e2 <= dx:
            e += dx
            pt0 = (pt0[0], pt0[1] + sy)
        yield pt0


class State:
    __actions = [(i - 1, j - 1) for i in range(3) for j in range(3)]

    def __init__(self, racetrack, position, velocity):
        self.__racetrack = racetrack
        self.__pos_x, self.__pos_y = position
        self.__vel_x, self.__vel_y = velocity

    @property
    def index(self):
       return (self.__pos_x, self.__pos_y, self.__vel_x, self.__vel_y)

    @property
    def position(self):
        return (self.__pos_x, self.__pos_y)
        
    @property
    def velocity(self):
        return (self.__vel_x, self.__vel_y)

    @property
    def actions(self):
        return [i for i, (ax, ay) in enumerate(self.__actions) if self.__vel_x + ax >= 0 and self.__vel_x + ax < 5 and self.__vel_y + ay >= 0 and self.__vel_y + ay < 5 and (self.__vel_x + ax != 0 or self.__vel_y + ay != 0)]

    def transition(self, action):
        ax, ay = self.__actions[action]
        vx = self.__vel_x + ax
        vy = self.__vel_y + ay
        if vx < 0 or vx >= 5 or vy < 0 or vy >= 5 or vx == 0 and vy == 0:
            raise IndexError("Invalid action")
        if random.random() < 0.1:
            vx = self.__vel_x
            vy = self.__vel_y
        px = self.__pos_x + vx
        py = self.__pos_y + vy
        trajectory = [p for p in bresenham_line((self.__pos_x, self.__pos_y), (px, py))]
        for x, y in trajectory[1:]:
            if x < 0 or x >= self.__racetrack.shape[0] or y < 0 or y >= self.__racetrack.shape[1]:
                return (State(self.__racetrack, random.choice(np.array(np.where(self.__racetrack == 1)).transpose().tolist()), (0, 0)), -1, trajectory)
            elif self.__racetrack[x, y] < 0:
                return (State(self.__racetrack, random.choice(np.array(np.where(self.__racetrack == 1)).transpose().tolist()), (0, 0)), -1, trajectory)
            elif self.__racetrack[x, y] == 2:
                return (None, 0, trajectory)
        return (State(self.__racetrack, (px, py), (vx, vy)), -1, trajectory)
