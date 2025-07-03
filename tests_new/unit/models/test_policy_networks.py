"""
Unit tests for policy networks.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from marl_autonomous_vehicle.models.policy_networks import (
    PolicyNetwork, LeaderPolicyNetwork, FollowerPolicyNetwork
)
from marl_autonomous_vehicle.utils import ActionSpace, Constants


class TestPolicyNetwork(unittest.TestCase):
    """Test policy network factory."""
    
    def test_create_leader_policy(self):
        """Test leader policy creation."""
        model = PolicyNetwork.create_leader_policy()
        self.assertIsNotNone(model)
    
    def test_create_follower_policy(self):
        """Test follower policy creation."""
        model = PolicyNetwork.create_follower_policy()
        self.assertIsNotNone(model)
    
    def test_create_leader_policy_custom_params(self):
        """Test leader policy with custom parameters."""
        model = PolicyNetwork.create_leader_policy(
            input_size=16,
            hidden_size=128,
            output_size=10
        )
        self.assertIsNotNone(model)
    
    def test_create_follower_policy_custom_params(self):
        """Test follower policy with custom parameters."""
        model = PolicyNetwork.create_follower_policy(
            input_size=16,
            hidden_size=128,
            output_size=10
        )
        self.assertIsNotNone(model)


class TestLeaderPolicyNetwork(unittest.TestCase):
    """Test leader policy network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = LeaderPolicyNetwork()
    
    def test_initialization(self):
        """Test policy network initialization."""
        self.assertIsNotNone(self.policy.model)
    
    def test_predict(self):
        """Test policy prediction."""
        # Mock observation
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        predictions = self.policy.predict(observation)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 1)  # Batch size 1
        self.assertEqual(len(predictions[0]), len(ActionSpace))  # Number of actions
    
    def test_predict_custom_size(self):
        """Test prediction with custom input size."""
        policy = LeaderPolicyNetwork(input_size=4, output_size=5)
        observation = [[1, 2, 3, 4]]
        
        predictions = policy.predict(observation)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions[0]), 5)  # Custom output size
    
    def test_callable(self):
        """Test that policy is callable."""
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        # Should be able to call like a function
        result = self.policy(observation)
        self.assertIsNotNone(result)


class TestFollowerPolicyNetwork(unittest.TestCase):
    """Test follower policy network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy = FollowerPolicyNetwork()
    
    def test_initialization(self):
        """Test policy network initialization."""
        self.assertIsNotNone(self.policy.model)
    
    def test_predict(self):
        """Test policy prediction."""
        # Mock observation
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        predictions = self.policy.predict(observation)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 1)  # Batch size 1
        self.assertEqual(len(predictions[0]), len(ActionSpace))  # Number of actions
    
    def test_predict_custom_size(self):
        """Test prediction with custom input size."""
        policy = FollowerPolicyNetwork(input_size=4, output_size=5)
        observation = [[1, 2, 3, 4]]
        
        predictions = policy.predict(observation)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions[0]), 5)  # Custom output size
    
    def test_callable(self):
        """Test that policy is callable."""
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        # Should be able to call like a function
        result = self.policy(observation)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()