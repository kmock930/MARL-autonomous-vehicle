"""
Wrapper for the SimpleGridEnv to provide a cleaner interface.
"""

import sys
import os
from typing import Dict, Tuple, List, Any, Optional
import numpy as np

# Add the gym-simplegrid to path
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)

try:
    from simple_grid import SimpleGridEnv
except ImportError:
    # Create a mock SimpleGridEnv for testing without dependencies
    class SimpleGridEnv:
        def __init__(self, **kwargs):
            self.env_configurations = kwargs
            self.obstacles = np.zeros((kwargs.get('rowSize', 10), kwargs.get('colSize', 10)))
            self.agents = []
            self.targets = []
            self.cumulative_reward = 0
            self.n_iter = 0
            self.OUT_OF_TETHER_COUNT = 0

        def reset(self, **kwargs):
            return {}, {}

        def step(self, actions, **kwargs):
            return {}, {}, False, False, {}

        def render(self):
            pass

from ..utils import ActionSpace, GridElements, Constants


class SimpleGridEnvWrapper:
    """
    Wrapper for SimpleGridEnv providing a cleaner, more modular interface.
    """

    def __init__(
        self,
        row_size: int = Constants.DEFAULT_GRID_SIZE,
        col_size: int = Constants.DEFAULT_GRID_SIZE,
        num_soft_obstacles: int = 10,
        num_hard_obstacles: int = 5,
        num_robots: int = 2,
        tether_dist: int = Constants.DEFAULT_TETHER_DISTANCE,
        num_leaders: int = 1,
        num_targets: int = 1,
        render_mode: str = "rgb_array"
    ):
        """
        Initialize the environment wrapper.
        
        Args:
            row_size: Number of rows in the grid
            col_size: Number of columns in the grid
            num_soft_obstacles: Number of soft obstacles
            num_hard_obstacles: Number of hard obstacles
            num_robots: Number of robots
            tether_dist: Maximum distance between robots
            num_leaders: Number of leader robots
            num_targets: Number of target locations
            render_mode: Rendering mode for the environment
        """
        self.env = SimpleGridEnv(
            render_mode=render_mode,
            rowSize=row_size,
            colSize=col_size,
            num_soft_obstacles=num_soft_obstacles,
            num_hard_obstacles=num_hard_obstacles,
            num_robots=num_robots,
            tetherDist=tether_dist,
            num_leaders=num_leaders,
            num_target=num_targets
        )
        
        self.row_size = row_size
        self.col_size = col_size
        self.tether_dist = tether_dist
        self.num_robots = num_robots
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observations, info)
        """
        if options is None:
            options = {}
        return self.env.reset(seed=seed, options=options)
    
    def step(self, actions: Dict[int, Tuple[int, int]], is_training: bool = False) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            is_training: Whether the environment is in training mode
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        return self.env.step(actions, isTraining=is_training)
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def get_observations(self) -> Dict:
        """Get current observations for all agents."""
        return self.env.get_obs()
    
    def get_info(self) -> Dict:
        """Get current environment information."""
        return self.env.get_info()
    
    @property
    def agents(self) -> List[Dict[str, Any]]:
        """Get list of agents in the environment."""
        return getattr(self.env, 'agents', [])
    
    @property
    def targets(self) -> List[Tuple[int, int]]:
        """Get list of target positions."""
        return getattr(self.env, 'targets', [])
    
    @property
    def obstacles(self) -> np.ndarray:
        """Get obstacle grid."""
        return getattr(self.env, 'obstacles', np.zeros((self.row_size, self.col_size)))
    
    @property
    def cumulative_reward(self) -> float:
        """Get cumulative reward."""
        return getattr(self.env, 'cumulative_reward', 0.0)
    
    @property
    def n_iter(self) -> int:
        """Get number of iterations."""
        return getattr(self.env, 'n_iter', 0)
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.row_size and 0 <= col < self.col_size
    
    def is_free(self, row: int, col: int) -> bool:
        """Check if position is free (not occupied by hard obstacle)."""
        if not self.is_in_bounds(row, col):
            return False
        return self.obstacles[row, col] != GridElements.OBSTACLE_HARD
    
    def compute_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Compute distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_valid_actions(self, agent_id: int) -> List[ActionSpace]:
        """
        Get valid actions for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of valid actions
        """
        if agent_id >= len(self.agents):
            return []
        
        agent_pos = self.agents[agent_id]['position']
        valid_actions = []
        
        for action in ActionSpace:
            new_pos = (
                agent_pos[0] + action.value[0],
                agent_pos[1] + action.value[1]
            )
            
            if self.is_in_bounds(new_pos[0], new_pos[1]) and self.is_free(new_pos[0], new_pos[1]):
                # Check if position is occupied by another agent
                occupied = any(
                    agent['position'] == new_pos 
                    for i, agent in enumerate(self.agents) 
                    if i != agent_id
                )
                
                if not occupied:
                    valid_actions.append(action)
        
        return valid_actions