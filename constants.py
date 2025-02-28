from enum import Enum

class ACTION_SPACE(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)
    STAY = (0, 0)

class REWARDS(Enum):
    SOFT_OBSTACLE = -10
    HARD_OBSTACLE = WALL = -50
    TARGET = 50
    STEP = -1
    CRASH = -50
    STAY = -3