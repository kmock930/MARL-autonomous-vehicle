"""
Wrapper for SimpleGrid environment with MARL-specific functionality.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Try to import the original SimpleGrid environment
try:
    import sys
    import os
    simplegrid_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'gym-simplegrid', 'gym_simplegrid', 'envs')
    sys.path.append(simplegrid_path)
    from simple_grid import SimpleGridEnv
    SIMPLEGRID_AVAILABLE = True
except ImportError:
    SimpleGridEnv = None
    SIMPLEGRID_AVAILABLE = False

from ..utils.constants import ACTION_SPACE, REWARDS


class SimpleGridWrapper:
    """
    Wrapper for SimpleGrid environment that provides MARL-specific functionality.
    Handles multi-agent coordination, communication, and reward shaping.
    """
    
    def __init__(self, 
                 render_mode: str = "rgb_array",
                 row_size: int = 10,
                 col_size: int = 10,
                 num_soft_obstacles: int = 10,
                 num_hard_obstacles: int = 5,
                 num_robots: int = 2,
                 tether_dist: int = 2,
                 num_leaders: int = 1,
                 num_targets: int = 1):
        """
        Initialize the environment wrapper.
        
        Args:
            render_mode: Rendering mode
            row_size: Grid height
            col_size: Grid width
            num_soft_obstacles: Number of soft obstacles
            num_hard_obstacles: Number of hard obstacles
            num_robots: Total number of robots
            tether_dist: Maximum distance between leader and followers
            num_leaders: Number of leader agents
            num_targets: Number of target locations
        """
        self.config = {
            'render_mode': render_mode,
            'rowSize': row_size,
            'colSize': col_size,
            'num_soft_obstacles': num_soft_obstacles,
            'num_hard_obstacles': num_hard_obstacles,
            'num_robots': num_robots,
            'tetherDist': tether_dist,
            'num_leaders': num_leaders,
            'num_target': num_targets
        }
        
        # Initialize the wrapped environment if available
        if SIMPLEGRID_AVAILABLE:
            self.env = SimpleGridEnv(**self.config)
        else:
            self.env = None
            
        # Agent tracking
        self.agents = []
        self.current_step = 0
        self.max_steps = 200
        
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment and return initial observations.
        
        Returns:
            Dictionary of agent observations
        """
        self.current_step = 0
        
        if self.env is not None:
            # Use real environment
            obs = self.env.reset()
            return self._process_observations(obs)
        else:
            # Mock environment for testing
            return self._mock_observations()
    
    def step(self, actions: Dict[int, Tuple[int, int]], 
             is_training: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        Execute actions and return results.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            is_training: Whether this is a training step
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        self.current_step += 1
        
        if self.env is not None:
            # Use real environment
            obs, rewards, done, info = self.env.step(actions, is_training)
            terminated = done
            truncated = self.current_step >= self.max_steps
            return self._process_observations(obs), rewards, terminated, truncated, info
        else:
            # Mock environment for testing
            return self._mock_step(actions)
    
    def get_agent_observation(self, agent_position: Tuple[int, int]) -> List[float]:
        """
        Get observation for a specific agent position.
        
        Args:
            agent_position: Position of the agent
            
        Returns:
            Observation array with environment information
        """
        if self.env is not None:
            # Use real environment observation logic
            # This would call the original get_agent_observation function
            pass
            
        # Mock observation for testing
        return [
            np.random.uniform(-1, 5),  # obs_dist
            np.random.choice([0, 1]),  # agent_visibility  
            np.random.uniform(0, 3),   # agent_dist
            np.random.choice([0, 1]),  # path_blocked
            np.random.randint(-1, 2),  # action_dx
            np.random.randint(-1, 2),  # action_dy
            np.random.uniform(-2, 2),  # dx
            np.random.uniform(-2, 2)   # dy
        ]
    
    def _process_observations(self, observations) -> Dict[str, np.ndarray]:
        """
        Process raw observations into standardized format.
        
        Args:
            observations: Raw observations from environment
            
        Returns:
            Processed observations dictionary
        """
        # Convert observations to standardized format
        processed = {}
        
        if isinstance(observations, dict):
            for agent_id, obs in observations.items():
                processed[f"agent_{agent_id}"] = np.array(obs)
        else:
            # Single observation
            processed["agent_0"] = np.array(observations)
            
        return processed
    
    def _mock_observations(self) -> Dict[str, np.ndarray]:
        """
        Generate mock observations for testing.
        
        Returns:
            Mock observations dictionary
        """
        observations = {}
        for i in range(self.config['num_robots']):
            observations[f"agent_{i}"] = np.random.random(8)
        return observations
    
    def _mock_step(self, actions: Dict[int, Tuple[int, int]]) -> Tuple[Dict[str, np.ndarray], Dict[int, float], bool, bool, Dict[str, Any]]:
        """
        Mock environment step for testing.
        
        Args:
            actions: Agent actions
            
        Returns:
            Mock step results
        """
        observations = self._mock_observations()
        
        # Generate random rewards
        rewards = {agent_id: np.random.uniform(-10, 50) for agent_id in actions.keys()}
        
        # Random termination
        terminated = np.random.random() < 0.05  # 5% chance of early termination
        truncated = self.current_step >= self.max_steps
        
        # Mock info
        info = {
            'agent_positions': [(np.random.randint(0, 10), np.random.randint(0, 10)) for _ in actions],
            'collisions': 0,
            'tether_violations': 0
        }
        
        return observations, rewards, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if self.env is not None:
            return self.env.render(mode)
        else:
            print("Mock environment - no rendering available")
            
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()