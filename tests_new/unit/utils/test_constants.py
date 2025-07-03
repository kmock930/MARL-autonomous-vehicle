"""
Unit tests for utility functions.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from marl_autonomous_vehicle.utils import ActionSpace, Rewards, Constants, GridElements


class TestConstants(unittest.TestCase):
    """Test constant definitions."""
    
    def test_action_space_values(self):
        """Test ActionSpace enum values."""
        self.assertEqual(ActionSpace.UP.value, (-1, 0))
        self.assertEqual(ActionSpace.DOWN.value, (1, 0))
        self.assertEqual(ActionSpace.LEFT.value, (0, -1))
        self.assertEqual(ActionSpace.RIGHT.value, (0, 1))
        self.assertEqual(ActionSpace.STAY.value, (0, 0))
    
    def test_reward_values(self):
        """Test Rewards enum values."""
        self.assertEqual(Rewards.TARGET.value, 50)
        self.assertEqual(Rewards.SOFT_OBSTACLE.value, -10)
        self.assertEqual(Rewards.HARD_OBSTACLE.value, -50)
        self.assertEqual(Rewards.STEP.value, -1)
    
    def test_grid_elements(self):
        """Test GridElements constants."""
        self.assertEqual(GridElements.FREE, 0)
        self.assertEqual(GridElements.OBSTACLE_SOFT, 1)
        self.assertEqual(GridElements.OBSTACLE_HARD, 2)
        self.assertEqual(GridElements.AGENT, 3)
        self.assertEqual(GridElements.TARGET, 4)
    
    def test_constants(self):
        """Test general constants."""
        self.assertEqual(Constants.LEADER_MESSAGE_SIZE, 8)
        self.assertEqual(Constants.TETHER_TOLERATE_COUNT, 5)
        self.assertEqual(Constants.DEFAULT_LEARNING_RATE, 0.001)


if __name__ == '__main__':
    unittest.main()