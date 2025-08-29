from enum import Enum

class ACTION_SPACE(Enum):
    # x: row, y: column
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    STAY = (0, 0)

class REWARDS(Enum):
    SOFT_OBSTACLE = -15
    HARD_OBSTACLE = -50
    WALL = -47
    TARGET = 250
    STEP = -1
    CRASH = OUT_OF_TETHER = -50
    STAY = -5

MESSAGE_SIZE = 1
AGENT_OBS_SIZE = 12
TETHER_TOLERATE_COUNT = 5
LEADER_MESSAGE_SIZE = 6 #Placeholder for Marl_5. TO be removed in final version
LEADER_OBS_SIZE = AGENT_OBS_SIZE
FOLLOWER_OBS_SIZE = AGENT_OBS_SIZE + MESSAGE_SIZE
SIGMA = 2