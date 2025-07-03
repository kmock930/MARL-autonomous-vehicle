"""
Integration tests for the MARL autonomous vehicle system.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from marl_autonomous_vehicle.utils import ActionSpace, Constants, generate_map
from marl_autonomous_vehicle.agents import LeaderAgent, FollowerAgent
from marl_autonomous_vehicle.models import PolicyNetwork, CriticNetwork, EncoderDecoder


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_map_generation_integration(self):
        """Test that map generation works with the system."""
        try:
            grid, robots, targets = generate_map(
                row_size=5,
                col_size=5,
                num_soft_obstacles=2,
                num_hard_obstacles=1,
                num_robots=2,
                tether_dist=2,
                num_targets=1
            )
            
            self.assertEqual(grid.shape, (5, 5))
            self.assertEqual(len(robots), 2)
            self.assertEqual(len(targets), 1)
            self.assertEqual(robots[0]['role'], 'leader')
            self.assertEqual(robots[1]['role'], 'follower')
        except ImportError:
            # Skip if numpy is not available
            self.skipTest("NumPy not available")
    
    def test_agent_communication_flow(self):
        """Test the communication flow between leader and follower."""
        # Create agents with mock models
        leader_policy = PolicyNetwork.create_leader_policy()
        follower_policy = PolicyNetwork.create_follower_policy()
        encoder_decoder = EncoderDecoder()
        
        leader = LeaderAgent(
            policy_model=leader_policy,
            encoder=encoder_decoder.get_encoder()
        )
        follower = FollowerAgent(
            policy_model=follower_policy,
            decoder=encoder_decoder.get_decoder()
        )
        
        # Test communication flow
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Leader acts and generates message
        leader_action = leader.act(observation)
        leader_message = leader.get_message()
        
        # Follower receives message and acts
        follower_action = follower.act(observation, leader_message)
        
        # Verify actions are valid
        self.assertIsInstance(leader_action, tuple)
        self.assertEqual(len(leader_action), 2)
        self.assertIsInstance(follower_action, tuple)
        self.assertEqual(len(follower_action), 2)
        
        # Verify message was generated and received
        self.assertIsNotNone(leader_message)
        self.assertIsNotNone(follower.get_decoded_message())
    
    def test_model_integration(self):
        """Test that all models can be created and work together."""
        # Create all model components
        leader_policy = PolicyNetwork.create_leader_policy()
        follower_policy = PolicyNetwork.create_follower_policy()
        critic = CriticNetwork()
        encoder_decoder = EncoderDecoder()
        
        # Test that models can be used
        observation = [[1, 2, 3, 4, 5, 6, 7, 8]]
        
        # Test policy networks
        leader_pred = leader_policy.predict(observation, verbose=0)
        follower_pred = follower_policy.predict(observation, verbose=0)
        
        # Test critic network
        value_pred = critic.predict(observation, verbose=0)
        
        # Test encoder-decoder
        encoded, decoded = encoder_decoder.encode_decode(observation, verbose=0)
        
        # Verify outputs have correct shapes
        self.assertEqual(len(leader_pred[0]), len(ActionSpace))
        self.assertEqual(len(follower_pred[0]), len(ActionSpace))
        self.assertEqual(len(value_pred[0]), 1)  # Single value output
        self.assertIsNotNone(encoded)
        self.assertIsNotNone(decoded)
    
    def test_agent_reset_and_state_management(self):
        """Test agent state management and reset functionality."""
        leader = LeaderAgent()
        follower = FollowerAgent()
        
        # Set initial state
        leader.set_position((1, 2))
        follower.set_position((3, 4))
        
        # Verify positions are set
        self.assertEqual(leader.get_position(), (1, 2))
        self.assertEqual(follower.get_position(), (3, 4))
        
        # Reset agents
        leader.reset()
        follower.reset()
        
        # Verify state is reset
        self.assertIsNone(leader.get_position())
        self.assertIsNone(follower.get_position())
        self.assertIsNone(leader.message)
        self.assertIsNone(follower.message)
    
    def test_action_space_integration(self):
        """Test that ActionSpace works correctly with agents."""
        leader = LeaderAgent()
        
        # Test all action space values are valid
        for action in ActionSpace:
            self.assertIsInstance(action.value, tuple)
            self.assertEqual(len(action.value), 2)
            
            # Test that action values are integers
            dx, dy = action.value
            self.assertIsInstance(dx, int)
            self.assertIsInstance(dy, int)
            
            # Test that actions are within reasonable bounds
            self.assertTrue(-1 <= dx <= 1)
            self.assertTrue(-1 <= dy <= 1)
    
    def test_constants_integration(self):
        """Test that constants are properly integrated across modules."""
        # Test that constants are accessible from different modules
        from marl_autonomous_vehicle.utils import Constants
        from marl_autonomous_vehicle.agents.base_agent import BaseAgent
        
        # Create an agent and verify it uses the constants
        leader = LeaderAgent()
        
        # Test observation normalization uses correct size
        observation = [1, 2, 3]  # Smaller than expected
        normalized = leader._normalize_observation(observation)
        
        self.assertEqual(normalized.shape[1], Constants.LEADER_MESSAGE_SIZE)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_missing_dependencies_graceful_handling(self):
        """Test that missing dependencies are handled gracefully."""
        # This test verifies that the system can still function 
        # even when optional dependencies are missing
        
        # Create agents without models
        leader = LeaderAgent()
        follower = FollowerAgent()
        
        # Should be able to act even without models
        observation = [1, 2, 3, 4, 5, 6, 7, 8]
        
        leader_action = leader.act(observation)
        follower_action = follower.act(observation)
        
        # Should return valid actions
        self.assertIsInstance(leader_action, tuple)
        self.assertIsInstance(follower_action, tuple)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        leader = LeaderAgent()
        
        # Test with empty observation
        action = leader.act([])
        self.assertIsInstance(action, tuple)
        
        # Test with very large observation
        large_obs = list(range(100))
        action = leader.act(large_obs)
        self.assertIsInstance(action, tuple)
        
        # Test with non-numeric observation (should handle gracefully)
        try:
            action = leader.act(['a', 'b', 'c'])
            self.assertIsInstance(action, tuple)
        except (ValueError, TypeError):
            # It's okay if it raises an error for invalid input
            pass


if __name__ == '__main__':
    unittest.main()