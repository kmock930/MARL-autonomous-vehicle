"""
Base agent class for the MARL autonomous vehicle system.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Dict
import numpy as np

from ..utils import ActionSpace, AgentRoles, Constants


class BaseAgent(ABC):
    """
    Base abstract class for agents in the MARL system.
    """
    
    _id_counter: int = 0
    
    def __init__(self, role: str):
        """
        Initialize a base agent.
        
        Args:
            role: Role of the agent ('leader' or 'follower')
        """
        self.role = role
        self.agent_id = BaseAgent._id_counter
        BaseAgent._id_counter += 1
        
        self.position: Optional[Tuple[int, int]] = None
        self.action_space = [action.value for action in ActionSpace]
        self.message: Optional[np.ndarray] = None
        
        # Model placeholders - will be set by subclasses
        self.policy_model = None
        self.encoder = None
        self.decoder = None
    
    @abstractmethod
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Choose an action based on observation and optional message.
        
        Args:
            observation: Environment observation
            message: Optional message from other agents
            
        Returns:
            Action as (dx, dy) tuple
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state."""
        self.position = None
        self.message = None
    
    def set_position(self, position: Tuple[int, int]) -> None:
        """Set agent position."""
        self.position = position
    
    def get_position(self) -> Optional[Tuple[int, int]]:
        """Get agent position."""
        return self.position
    
    def get_role(self) -> str:
        """Get agent role."""
        return self.role
    
    def get_id(self) -> int:
        """Get agent ID."""
        return self.agent_id
    
    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation to consistent format.
        
        Args:
            observation: Raw observation
            
        Returns:
            Normalized observation
        """
        # Flatten the grid observation to a 1D array
        observation = observation.flatten()

        # Ensure the observation has exactly LEADER_MESSAGE_SIZE elements
        if observation.size < Constants.LEADER_MESSAGE_SIZE:
            # Pad with zeros if the observation has fewer elements
            observation = np.pad(
                observation, 
                (0, Constants.LEADER_MESSAGE_SIZE - observation.size), 
                mode='constant'
            )
        elif observation.size > Constants.LEADER_MESSAGE_SIZE:
            # Truncate if the observation has more elements
            observation = observation[:Constants.LEADER_MESSAGE_SIZE]

        # Reshape observation to include batch dimension
        return observation.reshape(1, -1)
    
    def _get_action_from_predictions(self, predictions: np.ndarray) -> Tuple[int, int]:
        """
        Convert model predictions to action.
        
        Args:
            predictions: Model output predictions
            
        Returns:
            Action as (dx, dy) tuple
        """
        # Convert tensor to numpy if needed
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        # Get action with highest probability
        action_index = int(np.argmax(predictions))
        action_enum = list(ActionSpace)[action_index]
        
        return action_enum.value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role}, position={self.position})"