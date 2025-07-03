"""
Test utility constants and basic functionality.
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from marl_autonomous_vehicle.utils.constants import ACTION_SPACE, REWARDS, LEADER_MESSAGE_SIZE, TETHER_TOLERATE_COUNT


class TestConstants(unittest.TestCase):
    """Test constants and enumerations."""
    
    def test_action_space(self):
        """Test ACTION_SPACE enumeration."""
        # Test that all expected actions exist
        expected_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP_LEFT', 'UP_RIGHT', 'DOWN_LEFT', 'DOWN_RIGHT', 'STAY']
        for action_name in expected_actions:
            self.assertTrue(hasattr(ACTION_SPACE, action_name))
            
        # Test action values are tuples
        for action in ACTION_SPACE:
            self.assertIsInstance(action.value, tuple)
            self.assertEqual(len(action.value), 2)
            self.assertTrue(all(isinstance(val, int) for val in action.value))
            
        # Test specific action values
        self.assertEqual(ACTION_SPACE.UP.value, (-1, 0))
        self.assertEqual(ACTION_SPACE.DOWN.value, (1, 0))
        self.assertEqual(ACTION_SPACE.LEFT.value, (0, -1))
        self.assertEqual(ACTION_SPACE.RIGHT.value, (0, 1))
        self.assertEqual(ACTION_SPACE.STAY.value, (0, 0))
        
    def test_rewards(self):
        """Test REWARDS enumeration."""
        # Test that all expected rewards exist
        expected_rewards = ['SOFT_OBSTACLE', 'HARD_OBSTACLE', 'WALL', 'TARGET', 'STEP', 'CRASH', 'OUT_OF_TETHER', 'STAY']
        for reward_name in expected_rewards:
            self.assertTrue(hasattr(REWARDS, reward_name))
            
        # Test reward values are numbers
        for reward in REWARDS:
            self.assertIsInstance(reward.value, (int, float))
            
        # Test specific reward values
        self.assertEqual(REWARDS.SOFT_OBSTACLE.value, -10)
        self.assertEqual(REWARDS.HARD_OBSTACLE.value, -50)
        self.assertEqual(REWARDS.TARGET.value, 50)
        self.assertEqual(REWARDS.STEP.value, -1)
        
    def test_configuration_constants(self):
        """Test configuration constants."""
        self.assertEqual(LEADER_MESSAGE_SIZE, 8)
        self.assertIsInstance(LEADER_MESSAGE_SIZE, int)
        self.assertGreater(LEADER_MESSAGE_SIZE, 0)
        
        self.assertEqual(TETHER_TOLERATE_COUNT, 5)
        self.assertIsInstance(TETHER_TOLERATE_COUNT, int)
        self.assertGreater(TETHER_TOLERATE_COUNT, 0)
        
    def test_action_space_completeness(self):
        """Test that ACTION_SPACE covers all 9 possible grid movements."""
        # Should have 9 actions total (8 directions + stay)
        self.assertEqual(len(ACTION_SPACE), 9)
        
        # Collect all possible movements
        movements = set()
        for action in ACTION_SPACE:
            movements.add(action.value)
            
        # Should have 9 unique movements
        self.assertEqual(len(movements), 9)
        
        # Check that all combinations of -1, 0, 1 are covered
        expected_movements = {
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        }
        self.assertEqual(movements, expected_movements)


if __name__ == '__main__':
    unittest.main()