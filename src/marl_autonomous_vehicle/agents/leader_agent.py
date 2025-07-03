"""
Leader agent implementation.
"""

from typing import Tuple, Optional
import numpy as np

from .base_agent import BaseAgent
from ..utils import AgentRoles, Constants


class LeaderAgent(BaseAgent):
    """
    Leader agent that can send messages and make independent decisions.
    """
    
    def __init__(self, policy_model=None, encoder=None):
        """
        Initialize a leader agent.
        
        Args:
            policy_model: Neural network model for policy decisions
            encoder: Neural network model for encoding messages
        """
        super().__init__(AgentRoles.LEADER)
        self.policy_model = policy_model
        self.encoder = encoder
    
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Choose an action based on observation.
        
        Args:
            observation: Environment observation
            message: Not used by leader (for interface compatibility)
            
        Returns:
            Action as (dx, dy) tuple
        """
        # Normalize observation
        normalized_obs = self._normalize_observation(observation)
        
        # Generate message for followers
        self.message = self.speak(normalized_obs)
        
        # Make decision based on observation
        if self.policy_model is None:
            # Fallback to random action if no model
            return self._random_action()
        
        try:
            predictions = self.policy_model.predict(normalized_obs, verbose=0)
            return self._get_action_from_predictions(predictions)
        except Exception as e:
            print(f"Error in leader policy prediction: {e}")
            return self._random_action()
    
    def speak(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate a message to send to followers.
        
        Args:
            observation: Current observation (optional)
            
        Returns:
            Encoded message
        """
        if self.encoder is None:
            # Return dummy message if no encoder
            return np.zeros((1, Constants.LEADER_MESSAGE_SIZE))
        
        try:
            # Use observation or generate dummy input for encoder
            if observation is not None:
                input_data = observation
            else:
                # Use dummy input for testing
                import tensorflow as tf
                input_data = tf.random.normal((1, Constants.LEADER_MESSAGE_SIZE))
            
            encoded_message = self.encoder.predict(input_data, verbose=0)
            return encoded_message
        except Exception as e:
            print(f"Error in leader message encoding: {e}")
            return np.zeros((1, Constants.LEADER_MESSAGE_SIZE))
    
    def get_message(self) -> Optional[np.ndarray]:
        """
        Get the last generated message.
        
        Returns:
            Last encoded message or None
        """
        return self.message
    
    def _random_action(self) -> Tuple[int, int]:
        """Generate a random action as fallback."""
        from ..utils import ActionSpace
        import random
        action = random.choice(list(ActionSpace))
        return action.value