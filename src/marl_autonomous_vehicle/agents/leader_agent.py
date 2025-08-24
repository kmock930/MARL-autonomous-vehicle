"""
Leader agent implementation for the MARL autonomous vehicle system.
"""

import numpy as np
from typing import Tuple, Optional, Union, Any

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

from .base_agent import BaseAgent
from ..utils.constants import LEADER_MESSAGE_SIZE


class LeaderAgent(BaseAgent):
    """
    Leader agent responsible for navigation and communication with followers.
    Generates encoded messages to guide follower agents.
    """
    
    def __init__(self, agent_id: Optional[int] = None, 
                 encoder=None, policy_network=None):
        """
        Initialize a leader agent.
        
        Args:
            agent_id: Optional explicit agent ID
            encoder: Optional pre-trained encoder model
            policy_network: Optional pre-trained policy network
        """
        super().__init__("leader", agent_id)
        self.encoder = encoder
        self.policy_network = policy_network
        
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Leader decides action based on observation and generates message for followers.
        
        Args:
            observation: Partial observation of the environment
            message: Ignored for leader agents
            
        Returns:
            Action tuple (dx, dy)
        """
        # Prepare observation
        processed_obs = self._prepare_observation(observation)
        
        # Generate message for followers
        self.message = self.get_message()
        
        # Get action from policy network
        if self.policy_network is not None and TF_AVAILABLE:
            try:
                action_probs = self.policy_network.predict(processed_obs, verbose=0)
                return self._get_action_from_probs(action_probs)
            except Exception:
                # Fallback if model prediction fails
                pass
                
        # Fallback: simple policy based on observation
        return self._simple_policy(processed_obs)
    
    def get_message(self) -> np.ndarray:
        """
        Generate encoded message for follower agents.
        
        Returns:
            Encoded message array
        """
        if self.encoder is not None and TF_AVAILABLE:
            try:
                # Generate dummy input for encoder if needed
                dummy_input = tf.random.normal((1, LEADER_MESSAGE_SIZE))
                encoded_message = self.encoder.predict(dummy_input, verbose=0)
                return encoded_message
            except Exception:
                # Fallback if encoder fails
                pass
                
        # Fallback: return dummy message
        return np.zeros((1, 32))  # Standard encoded message size
    
    def speak(self) -> np.ndarray:
        """
        Alternative interface for getting encoded message.
        
        Returns:
            Encoded message array
        """
        return self.get_message()
        
    def _simple_policy(self, observation: np.ndarray) -> Tuple[int, int]:
        """
        Simple fallback policy when neural network is unavailable.
        
        Args:
            observation: Processed observation
            
        Returns:
            Action tuple (dx, dy)
        """
        # Simple heuristic: prefer moving right and down
        if len(observation.flatten()) >= 2:
            obs_flat = observation.flatten()
            if obs_flat[0] > 0.5:  # Some obstacle ahead
                return (0, 1)  # Move right
            elif obs_flat[1] > 0.5:  # Some obstacle to the right
                return (1, 0)  # Move down
                
        # Default: stay in place
        return (0, 0)