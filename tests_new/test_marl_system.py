"""
Test complete MARL system integration.
"""

import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from marl_autonomous_vehicle import LeaderAgent, FollowerAgent, MAPPOTrainer, TrainingConfig, SimpleGridWrapper
from marl_autonomous_vehicle.models import PolicyNetwork, EncoderDecoder
import numpy as np


class TestMARLSystem(unittest.TestCase):
    """Test the complete MARL system integration."""
    
    def setUp(self):
        """Set up the system components."""
        self.config = TrainingConfig(episodes=5)  # Short test
        self.leader = LeaderAgent()
        self.follower = FollowerAgent()
        self.environment = SimpleGridWrapper()
        
    def test_system_creation(self):
        """Test that all system components can be created."""
        # Test component creation
        self.assertIsNotNone(self.config)
        self.assertIsNotNone(self.leader)
        self.assertIsNotNone(self.follower)
        self.assertIsNotNone(self.environment)
        
        # Test trainer creation
        trainer = MAPPOTrainer(
            config=self.config,
            leader_agent=self.leader,
            follower_agent=self.follower,
            environment=self.environment
        )
        self.assertIsNotNone(trainer)
        
    def test_training_workflow(self):
        """Test complete training workflow without ML dependencies."""
        trainer = MAPPOTrainer(
            config=self.config,
            leader_agent=self.leader,
            follower_agent=self.follower,
            environment=self.environment
        )
        
        # Run training (mock)
        history = trainer.train()
        
        # Verify training history structure
        self.assertIsInstance(history, dict)
        expected_keys = ['policy_loss', 'value_loss', 'contrastive_loss', 
                        'reconstruction_loss', 'total_loss', 'rewards', 'episode_lengths']
        for key in expected_keys:
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), self.config.episodes)
            
    def test_environment_interaction(self):
        """Test agent-environment interaction."""
        # Reset environment
        observations = self.environment.reset()
        self.assertIsInstance(observations, dict)
        
        # Agents act
        obs_array = list(observations.values())[0]
        leader_action = self.leader.act(obs_array)
        leader_message = self.leader.get_message()
        follower_action = self.follower.act(obs_array, leader_message)
        
        # Execute step
        actions = {0: leader_action, 1: follower_action}
        step_result = self.environment.step(actions)
        
        # Verify step result structure
        new_obs, rewards, terminated, truncated, info = step_result
        self.assertIsInstance(new_obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        
    def test_model_factories(self):
        """Test model factory methods."""
        # Test policy network creation (should return None without TensorFlow)
        leader_policy = PolicyNetwork.create_leader_policy()
        follower_policy = PolicyNetwork.create_follower_policy()
        critic = PolicyNetwork.create_critic_network()
        
        # Without TensorFlow, these should be None
        # But the factory methods should not crash
        
        # Test encoder-decoder creation
        encoder, decoder = EncoderDecoder.create_encoder_decoder_pair()
        
        # These should also be None without TensorFlow
        # But creation should not fail
        
    def test_configuration_management(self):
        """Test training configuration management."""
        # Test default config
        config = TrainingConfig()
        self.assertEqual(config.episodes, 1000)
        self.assertEqual(config.learning_rate, 0.001)
        
        # Test config modification
        new_config = config.update(episodes=500, learning_rate=0.01)
        self.assertEqual(new_config.episodes, 500)
        self.assertEqual(new_config.learning_rate, 0.01)
        self.assertEqual(config.episodes, 1000)  # Original unchanged
        
        # Test serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('episodes', config_dict)
        
        # Test deserialization
        restored_config = TrainingConfig.from_dict(config_dict)
        self.assertEqual(restored_config.episodes, config.episodes)
        
    def test_backward_compatibility(self):
        """Test that the system maintains backward compatibility."""
        # Test that main classes can be imported from package root
        from marl_autonomous_vehicle import LeaderAgent as LA
        from marl_autonomous_vehicle import FollowerAgent as FA
        from marl_autonomous_vehicle import ACTION_SPACE, LEADER_MESSAGE_SIZE
        
        # Create agents using imported classes
        leader = LA()
        follower = FA()
        
        self.assertEqual(leader.role, "leader")
        self.assertEqual(follower.role, "follower")
        
        # Test constants
        self.assertIsNotNone(ACTION_SPACE)
        self.assertEqual(LEADER_MESSAGE_SIZE, 8)
        
    def test_error_handling(self):
        """Test graceful error handling."""
        # Test with invalid observations
        try:
            leader_action = self.leader.act(None)
            # Should handle None gracefully or raise appropriate error
        except Exception as e:
            # Should be a reasonable error message
            self.assertIsInstance(e, (TypeError, AttributeError))
            
        # Test training with missing components
        minimal_trainer = MAPPOTrainer(
            config=self.config,
            leader_agent=self.leader,
            follower_agent=self.follower
        )
        
        # Should still be able to train (in mock mode)
        history = minimal_trainer.train()
        self.assertIsNotNone(history)


if __name__ == '__main__':
    unittest.main()