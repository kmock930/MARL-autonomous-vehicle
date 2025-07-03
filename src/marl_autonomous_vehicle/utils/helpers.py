"""
Helper functions and utilities for the MARL autonomous vehicle system.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Any
from enum import Enum


# Define constants locally to avoid circular imports
class GridElements:
    """Constants for grid elements."""
    FREE = 0
    OBSTACLE_SOFT = 1
    OBSTACLE_HARD = 2
    AGENT = 3
    TARGET = 4


class ActionSpace(Enum):
    """Action space for agents in the grid environment."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    STAY = (0, 0)


def generate_map(
    row_size: int,
    col_size: int,
    num_soft_obstacles: int,
    num_hard_obstacles: int,
    num_robots: int,
    tether_dist: int,
    num_leaders: int = 1,
    num_targets: int = 1
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int]]]:
    """
    Generate a random map with obstacles, robots, and targets.
    
    Args:
        row_size: Number of rows in the grid
        col_size: Number of columns in the grid
        num_soft_obstacles: Number of soft obstacles to place
        num_hard_obstacles: Number of hard obstacles to place
        num_robots: Number of robots to place
        tether_dist: Maximum distance between robots
        num_leaders: Number of leader robots
        num_targets: Number of target locations
        
    Returns:
        Tuple containing the grid, robot list, and target list
        
    Raises:
        ValueError: If there are too many objects for the grid size
    """
    # Initialize the map with free cells
    grid = np.zeros((row_size, col_size), dtype=int)

    total_cells = row_size * col_size
    total_obstacles = num_soft_obstacles + num_hard_obstacles
    if (total_obstacles + num_robots + num_targets) > total_cells:
        raise ValueError("Total number of obstacles, robots, and targets exceeds the grid size.")

    # Place soft obstacles randomly
    for _ in range(num_soft_obstacles):
        while True:
            x, y = random.randint(0, row_size-1), random.randint(0, col_size-1)
            if grid[x, y] == GridElements.FREE:
                grid[x, y] = GridElements.OBSTACLE_SOFT
                break

    # Place hard obstacles randomly
    for _ in range(num_hard_obstacles):
        while True:
            x, y = random.randint(0, row_size-1), random.randint(0, col_size-1)
            if grid[x, y] == GridElements.FREE:
                grid[x, y] = GridElements.OBSTACLE_HARD
                break

    # Place robots randomly with roles
    robots = []
    roles = ['leader', 'follower']
    for i in range(num_robots):
        while True:
            x, y = random.randint(0, row_size-1), random.randint(0, col_size-1)
            if grid[x, y] == GridElements.FREE:
                if i == 0 or all(max(abs(x - robot['position'][0]), abs(y - robot['position'][1])) <= tether_dist for robot in robots):
                    grid[x, y] = GridElements.AGENT
                    robots.append({
                        'position': (x, y), 
                        'role': roles[0 if i < num_leaders else 1]
                    })
                    break

    # Place the targets randomly
    targets = []
    for i in range(num_targets):
        while True:
            x, y = random.randint(0, row_size-1), random.randint(0, col_size-1)
            if grid[x, y] == GridElements.FREE:
                grid[x, y] = GridElements.TARGET
                targets.append((x, y))
                break

    return grid, robots, targets


def print_map(grid: np.ndarray) -> None:
    """Print the grid map in a readable format."""
    for row in grid:
        print(' '.join(str(cell) for cell in row))


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def is_valid_position(
    position: Tuple[int, int],
    grid_shape: Tuple[int, int],
    obstacles: np.ndarray,
    agents: List[Dict[str, Any]]
) -> bool:
    """
    Check if a position is valid (within bounds, not occupied, not a hard obstacle).
    
    Args:
        position: (x, y) position to check
        grid_shape: Shape of the grid (rows, cols)
        obstacles: Obstacle grid
        agents: List of agent dictionaries
        
    Returns:
        True if position is valid, False otherwise
    """
    x, y = position
    rows, cols = grid_shape
    
    # Check bounds
    if not (0 <= x < rows and 0 <= y < cols):
        return False
    
    # Check for agent collision
    for agent in agents:
        if agent["position"] == position:
            return False
    
    # Check for hard obstacle
    if obstacles[x, y] == GridElements.OBSTACLE_HARD:
        return False
    
    return True


def get_valid_actions(
    agent_position: Tuple[int, int],
    grid_shape: Tuple[int, int],
    obstacles: np.ndarray,
    agents: List[Dict[str, Any]]
) -> List[ActionSpace]:
    """
    Get list of valid actions for an agent at a given position.
    
    Args:
        agent_position: Current position of the agent
        grid_shape: Shape of the grid
        obstacles: Obstacle grid
        agents: List of all agents
        
    Returns:
        List of valid ActionSpace values
    """
    valid_actions = []
    
    for action in ActionSpace:
        new_pos = (
            agent_position[0] + action.value[0],
            agent_position[1] + action.value[1]
        )
        
        if is_valid_position(new_pos, grid_shape, obstacles, agents):
            valid_actions.append(action)
    
    return valid_actions


def normalize_observation(observation: np.ndarray, max_size: int = 8) -> np.ndarray:
    """
    Normalize observation to a fixed size for consistent model input.
    
    Args:
        observation: Raw observation array
        max_size: Target size for normalization
        
    Returns:
        Normalized observation array
    """
    # Flatten the observation
    obs_flat = observation.flatten()
    
    # Ensure the observation has exactly max_size elements
    if obs_flat.size < max_size:
        # Pad with zeros if the observation has fewer than max_size elements
        obs_flat = np.pad(obs_flat, (0, max_size - obs_flat.size), mode='constant')
    elif obs_flat.size > max_size:
        # Truncate if the observation has more than max_size elements
        obs_flat = obs_flat[:max_size]
    
    return obs_flat.reshape(1, -1)