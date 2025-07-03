"""
Follower agent implementation.
"""

from typing import Tuple, Optional
import numpy as np

from .base_agent import BaseAgent
from ..utils import AgentRoles, Constants


class FollowerAgent(BaseAgent):
    """
    Follower agent that receives messages and makes decisions based on them.
    """
    
    def __init__(self, policy_model=None, decoder=None):
        """
        Initialize a follower agent.
        
        Args:
            policy_model: Neural network model for policy decisions
            decoder: Neural network model for decoding messages
        """
        super().__init__(AgentRoles.FOLLOWER)
        self.policy_model = policy_model
        self.decoder = decoder
    
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Choose an action based on observation and leader message.
        
        Args:
            observation: Environment observation
            message: Message from leader agent
            
        Returns:
            Action as (dx, dy) tuple
        """
        # Normalize observation
        normalized_obs = self._normalize_observation(observation)
        
        # Process message from leader
        if message is not None:
            self.listen(message)
        else:
            # Empty message
            self.message = np.zeros((1, 32))  # Default decoded message size
        
        # Make decision based on observation and message
        if self.policy_model is None:
            # Fallback to random action if no model
            return self._random_action()
        
        try:
            # Combine observation and decoded message
            combined_input = np.concatenate((normalized_obs, self.message), axis=1)
            # Adjust to match expected input shape
            combined_input = combined_input[:, :Constants.LEADER_MESSAGE_SIZE]
            
            predictions = self.policy_model.predict(combined_input, verbose=0)
            return self._get_action_from_predictions(predictions)
        except Exception as e:
            print(f"Error in follower policy prediction: {e}")
            return self._random_action()
    
    def listen(self, message: np.ndarray) -> None:
        """
        Process a message from the leader.
        
        Args:
            message: Encoded message from leader
        """
        if self.decoder is None:
            # If no decoder, use message as-is (limited to message size)
            if message.shape[1] > Constants.LEADER_MESSAGE_SIZE:
                self.message = message[:, :Constants.LEADER_MESSAGE_SIZE]
            else:
                # Pad if necessary
                padding_size = Constants.LEADER_MESSAGE_SIZE - message.shape[1]
                self.message = np.pad(message, ((0, 0), (0, padding_size)), mode='constant')
            return
        
        try:
            self.message = self.decoder.predict(message, verbose=0)
        except Exception as e:
            print(f"Error in message decoding: {e}")
            self.message = np.zeros((1, Constants.LEADER_MESSAGE_SIZE))
    
    def get_decoded_message(self) -> Optional[np.ndarray]:
        """
        Get the last decoded message.
        
        Returns:
            Last decoded message or None
        """
        return self.message
    
    def _random_action(self) -> Tuple[int, int]:
        """Generate a random action as fallback."""
        from ..utils import ActionSpace
        import random
        action = random.choice(list(ActionSpace))
        return action.value