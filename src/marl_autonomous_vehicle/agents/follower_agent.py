"""
Follower agent implementation for the MARL autonomous vehicle system.
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


class FollowerAgent(BaseAgent):
    """
    Follower agent that acts based on its own observation and messages from the leader.
    Uses a decoder to interpret leader messages and combines them with local observations.
    """
    
    def __init__(self, agent_id: Optional[int] = None, 
                 decoder=None, policy_network=None):
        """
        Initialize a follower agent.
        
        Args:
            agent_id: Optional explicit agent ID
            decoder: Optional pre-trained decoder model
            policy_network: Optional pre-trained policy network
        """
        super().__init__("follower", agent_id)
        self.decoder = decoder
        self.policy_network = policy_network
        
    def act(self, observation: np.ndarray, message: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Follower decides action based on observation and leader message.
        
        Args:
            observation: Partial observation of the environment
            message: Encoded message from leader agent
            
        Returns:
            Action tuple (dx, dy)
        """
        # Prepare observation
        processed_obs = self._prepare_observation(observation)
        
        # Process leader message
        if message is None:
            # Empty message if no leader communication
            self.message = np.zeros((1, 32))
        else:
            self.listen(message)
            
        # Get action from policy network
        if self.policy_network is not None and TF_AVAILABLE:
            try:
                # Combine observation with decoded message
                combined_input = self._combine_inputs(processed_obs, self.message)
                action_probs = self.policy_network.predict(combined_input, verbose=0)
                return self._get_action_from_probs(action_probs)
            except Exception:
                # Fallback if model prediction fails
                pass
                
        # Fallback: simple policy based on observation
        return self._simple_policy(processed_obs)
    
    def listen(self, message: np.ndarray) -> None:
        """
        Decode and store message from leader agent.
        
        Args:
            message: Encoded message from leader
        """
        if self.decoder is not None and TF_AVAILABLE:
            try:
                self.message = self.decoder.predict(message, verbose=0)
                return
            except Exception:
                # Fallback if decoder fails
                pass
                
        # Fallback: use message as-is or create dummy
        if message is not None:
            self.message = message
        else:
            self.message = np.zeros((1, 32))
            
    def _combine_inputs(self, observation: np.ndarray, message: np.ndarray) -> np.ndarray:
        """
        Combine observation with decoded message for policy input.
        
        Args:
            observation: Processed observation
            message: Decoded leader message
            
        Returns:
            Combined input array
        """
        try:
            # Ensure both inputs have compatible shapes
            obs_flat = observation.flatten()
            msg_flat = message.flatten()
            
            # Combine inputs
            combined = np.concatenate([obs_flat, msg_flat])
            
            # Adjust to match expected input shape (truncate if too long)
            if len(combined) > LEADER_MESSAGE_SIZE:
                combined = combined[:LEADER_MESSAGE_SIZE]
            elif len(combined) < LEADER_MESSAGE_SIZE:
                # Pad if too short
                combined = np.pad(combined, (0, LEADER_MESSAGE_SIZE - len(combined)), mode='constant')
                
            return combined.reshape(1, -1)
            
        except Exception:
            # Fallback: return observation only
            return observation
            
    def _simple_policy(self, observation: np.ndarray) -> Tuple[int, int]:
        """
        Simple fallback policy when neural network is unavailable.
        
        Args:
            observation: Processed observation
            
        Returns:
            Action tuple (dx, dy)
        """
        # Simple heuristic: try to follow some basic rules
        if len(observation.flatten()) >= 3:
            obs_flat = observation.flatten()
            if obs_flat[0] < 0.3:  # Clear path ahead
                return (-1, 0)  # Move up (towards leader typically)
            elif obs_flat[2] < 0.3:  # Clear path to the right
                return (0, 1)  # Move right
                
        # Default: stay in place
        return (0, 0)