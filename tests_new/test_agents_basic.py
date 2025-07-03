"""
Test basic agent functionality without external dependencies.
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from marl_autonomous_vehicle.agents import LeaderAgent, FollowerAgent
from marl_autonomous_vehicle.utils import ACTION_SPACE, LEADER_MESSAGE_SIZE
import numpy as np


class TestAgentsBasic(unittest.TestCase):
    """Test basic agent functionality without ML dependencies."""
    
    def setUp(self):
        """Set up test agents."""
        self.leader = LeaderAgent()
        self.follower = FollowerAgent()
        
    def test_agent_creation(self):
        """Test that agents can be created successfully."""
        self.assertEqual(self.leader.role, "leader")
        self.assertEqual(self.follower.role, "follower")
        self.assertIsInstance(self.leader.agent_id, int)
        self.assertIsInstance(self.follower.agent_id, int)
        
    def test_agent_action(self):
        """Test that agents can take actions."""
        observation = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Test leader action
        leader_action = self.leader.act(observation)
        self.assertIsInstance(leader_action, tuple)
        self.assertEqual(len(leader_action), 2)
        self.assertTrue(all(isinstance(val, int) for val in leader_action))
        
        # Test follower action
        follower_action = self.follower.act(observation)
        self.assertIsInstance(follower_action, tuple)
        self.assertEqual(len(follower_action), 2)
        self.assertTrue(all(isinstance(val, int) for val in follower_action))
        
    def test_leader_message(self):
        """Test that leader can generate messages."""
        message = self.leader.get_message()
        self.assertIsInstance(message, np.ndarray)
        self.assertEqual(len(message.shape), 2)  # Should be 2D array
        
    def test_follower_listen(self):
        """Test that follower can process messages."""
        # Create a dummy message
        message = np.random.random((1, 32))
        
        # Follower should be able to listen to message
        self.follower.listen(message)
        self.assertIsNotNone(self.follower.message)
        
    def test_communication_flow(self):
        """Test leader-follower communication without dependencies."""
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Leader acts and generates message
        leader_action = self.leader.act(observation)
        message = self.leader.get_message()
        
        # Follower acts with leader's message
        follower_action = self.follower.act(observation, message)
        
        # Verify valid actions returned
        self.assertIsInstance(leader_action, tuple)
        self.assertIsInstance(follower_action, tuple)
        
        # Verify actions are valid movement tuples
        for action in [leader_action, follower_action]:
            self.assertEqual(len(action), 2)
            self.assertTrue(all(isinstance(val, int) for val in action))
            # Check that action is within reasonable bounds
            self.assertTrue(all(-1 <= val <= 1 for val in action))


if __name__ == '__main__':
    unittest.main()