"""
Unit tests for leader agent functionality.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from marl_autonomous_vehicle.agents.leader_agent import LeaderAgent
from marl_autonomous_vehicle.utils import AgentRoles, Constants


class TestLeaderAgent(unittest.TestCase):
    """Test leader agent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_policy = Mock()
        self.mock_encoder = Mock()
        
        # Configure mock returns
        self.mock_policy.predict.return_value = [[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.mock_encoder.predict.return_value = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        self.agent = LeaderAgent(
            policy_model=self.mock_policy,
            encoder=self.mock_encoder
        )
    
    def test_initialization(self):
        """Test leader agent initialization."""
        self.assertEqual(self.agent.role, AgentRoles.LEADER)
        self.assertEqual(self.agent.policy_model, self.mock_policy)
        self.assertEqual(self.agent.encoder, self.mock_encoder)
        self.assertIsNotNone(self.agent.agent_id)
    
    def test_initialization_without_models(self):
        """Test initialization without models."""
        agent = LeaderAgent()
        self.assertEqual(agent.role, AgentRoles.LEADER)
        self.assertIsNone(agent.policy_model)
        self.assertIsNone(agent.encoder)
    
    def test_act_with_model(self):
        """Test action selection with policy model."""
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        action = self.agent.act(observation)
        
        # Should return the action with highest probability (index 1 = DOWN)
        self.assertEqual(action, (1, 0))  # DOWN action
        self.mock_policy.predict.assert_called_once()
        self.assertIsNotNone(self.agent.message)  # Should have generated a message
    
    def test_act_without_model(self):
        """Test action selection without policy model."""
        agent = LeaderAgent()
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        action = agent.act(observation)
        
        # Should return a valid action (any action from ActionSpace)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)
    
    def test_speak_with_encoder(self):
        """Test message generation with encoder."""
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        message = self.agent.speak(observation)
        
        self.assertIsNotNone(message)
        self.mock_encoder.predict.assert_called_once()
    
    def test_speak_without_encoder(self):
        """Test message generation without encoder."""
        agent = LeaderAgent()
        
        message = agent.speak()
        
        self.assertIsNotNone(message)
        self.assertEqual(message.shape, (1, Constants.LEADER_MESSAGE_SIZE))
    
    def test_speak_without_observation(self):
        """Test message generation without observation."""
        with patch('tensorflow.random.normal') as mock_random:
            mock_random.return_value = [[0, 1, 2, 3, 4, 5, 6, 7]]
            
            message = self.agent.speak()
            
            self.assertIsNotNone(message)
    
    def test_get_message(self):
        """Test getting the last generated message."""
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Act to generate a message
        self.agent.act(observation)
        
        message = self.agent.get_message()
        self.assertIsNotNone(message)
    
    def test_get_message_none(self):
        """Test getting message when none has been generated."""
        message = self.agent.get_message()
        self.assertIsNone(message)
    
    def test_observation_normalization(self):
        """Test that observations are properly normalized."""
        # Test with smaller observation
        small_obs = [1, 2, 3]
        action = self.agent.act(small_obs)
        
        # Should still work and call predict with normalized observation
        self.mock_policy.predict.assert_called()
        call_args = self.mock_policy.predict.call_args[0][0]
        self.assertEqual(call_args.shape, (1, Constants.LEADER_MESSAGE_SIZE))
    
    def test_error_handling_in_act(self):
        """Test error handling in act method."""
        # Make policy prediction raise an exception
        self.mock_policy.predict.side_effect = Exception("Model error")
        
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        action = self.agent.act(observation)
        
        # Should still return a valid action (fallback)
        self.assertIsInstance(action, tuple)
        self.assertEqual(len(action), 2)
    
    def test_error_handling_in_speak(self):
        """Test error handling in speak method."""
        # Make encoder raise an exception
        self.mock_encoder.predict.side_effect = Exception("Encoder error")
        
        message = self.agent.speak()
        
        # Should return a default message
        self.assertIsNotNone(message)
        self.assertEqual(message.shape, (1, Constants.LEADER_MESSAGE_SIZE))


if __name__ == '__main__':
    unittest.main()