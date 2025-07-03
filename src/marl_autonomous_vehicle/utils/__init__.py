"""
Constants and enums for the MARL autonomous vehicle system.
"""

from enum import Enum
from typing import Final


class ActionSpace(Enum):
    """Action space for agents in the grid environment."""
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


class Rewards(Enum):
    """Reward values for different actions and outcomes."""
    SOFT_OBSTACLE = -10
    HARD_OBSTACLE = -50
    WALL = -47
    TARGET = 50
    STEP = -1
    CRASH = -50
    OUT_OF_TETHER = -50
    STAY = -3


class GridElements:
    """Constants for grid elements."""
    FREE: Final[int] = 0
    OBSTACLE_SOFT: Final[int] = 1
    OBSTACLE_HARD: Final[int] = 2
    AGENT: Final[int] = 3
    TARGET: Final[int] = 4


class AgentRoles:
    """Agent role constants."""
    LEADER: Final[str] = "leader"
    FOLLOWER: Final[str] = "follower"


class Constants:
    """General constants for the system."""
    LEADER_MESSAGE_SIZE: Final[int] = 8
    TETHER_TOLERATE_COUNT: Final[int] = 5
    DEFAULT_LEARNING_RATE: Final[float] = 0.001
    DEFAULT_BATCH_SIZE: Final[int] = 32
    DEFAULT_GRID_SIZE: Final[int] = 10
    DEFAULT_TETHER_DISTANCE: Final[int] = 2


# Import helper functions conditionally
try:
    from .helpers import (
        generate_map,
        print_map,
        calculate_distance,
        is_valid_position,
        get_valid_actions,
        normalize_observation
    )
    
    # Export everything for easy access
    __all__ = [
        "ActionSpace", "Rewards", "GridElements", "AgentRoles", "Constants",
        "generate_map", "print_map", "calculate_distance", "is_valid_position",
        "get_valid_actions", "normalize_observation"
    ]
except ImportError:
    # Export just the constants if numpy is not available
    __all__ = ["ActionSpace", "Rewards", "GridElements", "AgentRoles", "Constants"]