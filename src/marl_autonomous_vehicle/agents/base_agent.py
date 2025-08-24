"""
Base agent class providing common functionality for all agent types.
"""

import numpy as np
from typing import Tuple, Union, Optional, Any, List
from abc import ABC, abstractmethod

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

from ..utils.constants import ACTION_SPACE, LEADER_MESSAGE_SIZE


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the MARL system.
    Provides common functionality and enforces interface consistency.
    """
    
    _id_counter = 0  # Class-level counter for auto-assigning agent IDs

    def __init__(self, role: str, agent_id: Optional[int] = None):
        """
        Initialize a base agent.
        
        Args:
            role: String indicator of the agent's role ("leader" or "follower")
            agent_id: Optional explicit agent ID, auto-assigned if None
        """
        self.role = role
        self.action_space = [action.value for action in ACTION_SPACE]
        
        # Auto-assign or use provided agent ID
        if agent_id is None:
            self.agent_id = BaseAgent._id_counter
            BaseAgent._id_counter += 1
        else:
            self.agent_id = agent_id
            
        # Initialize message storage
        self.message: Optional[Union[np.ndarray, Any]] = None
        
        # Initialize models - will be set by subclasses or dependency injection
        self.encoder = None
        self.decoder = None
        self.policy_network = None
        
    @abstractmethod
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Take an action based on observation and optional message.
        
        Args:
            observation: Partial observation of the environment
            message: Optional message from other agents
            
        Returns:
            Action tuple (dx, dy)
        """
        pass
    
    def reset(self):
        """Reset agent state for new episode."""
        self.message = None
        
    def _prepare_observation(self, observation: Union[np.ndarray, list]) -> np.ndarray:
        """
        Prepare observation by ensuring correct shape and size.
        
        Args:
            observation: Raw observation array or list
            
        Returns:
            Processed observation with correct dimensions
        """
        # Convert to numpy array if it's a list
        if isinstance(observation, list):
            observation = np.array(observation)
            
        # Flatten the grid observation to a 1D array
        observation = observation.flatten()

        # Ensure the observation has exactly LEADER_MESSAGE_SIZE elements
        if observation.size < LEADER_MESSAGE_SIZE:
            # Pad with zeros if the observation has fewer than required elements
            observation = np.pad(
                observation, 
                (0, LEADER_MESSAGE_SIZE - observation.size), 
                mode='constant'
            )
        elif observation.size > LEADER_MESSAGE_SIZE:
            # Truncate if the observation has more than required elements
            observation = observation[:LEADER_MESSAGE_SIZE]

        # Reshape observation to include batch dimension
        return observation.reshape(1, -1)
        
    def _get_action_from_probs(self, action_probs: Union[np.ndarray, Any]) -> Tuple[int, int]:
        """
        Convert action probabilities to action tuple.
        
        Args:
            action_probs: Action probability array
            
        Returns:
            Action tuple (dx, dy)
        """
        # Handle TensorFlow tensors if available
        if TF_AVAILABLE and hasattr(action_probs, 'numpy'):
            action_probs = action_probs.numpy()
            
        # Get action with highest probability
        action_index = int(np.argmax(action_probs))
        return list(ACTION_SPACE)[action_index].value
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role='{self.role}')"