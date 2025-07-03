"""
Unit tests for base agent functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.marl_autonomous_vehicle.agents.base_agent import BaseAgent
from src.marl_autonomous_vehicle.utils import ActionSpace, Constants


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def act(self, observation, message=None):
        return (0, 1)  # Simple test action


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent("leader")
        
        assert agent.role == "leader"
        assert agent.agent_id >= 0
        assert agent.position is None
        assert agent.message is None
        assert len(agent.action_space) == len(ActionSpace)
    
    def test_agent_id_increment(self):
        """Test that agent IDs increment properly."""
        initial_counter = BaseAgent._id_counter
        agent1 = ConcreteAgent("leader")
        agent2 = ConcreteAgent("follower")
        
        assert agent2.agent_id == agent1.agent_id + 1
    
    def test_set_and_get_position(self):
        """Test setting and getting agent position."""
        agent = ConcreteAgent("leader")
        position = (3, 4)
        
        agent.set_position(position)
        assert agent.get_position() == position
        assert agent.position == position
    
    def test_get_role(self):
        """Test getting agent role."""
        agent = ConcreteAgent("follower")
        assert agent.get_role() == "follower"
    
    def test_get_id(self):
        """Test getting agent ID."""
        agent = ConcreteAgent("leader")
        assert agent.get_id() == agent.agent_id
    
    def test_reset(self):
        """Test resetting agent state."""
        agent = ConcreteAgent("leader")
        agent.set_position((1, 2))
        agent.message = np.array([1, 2, 3])
        
        agent.reset()
        
        assert agent.position is None
        assert agent.message is None
    
    def test_normalize_observation_exact_size(self):
        """Test observation normalization with exact size."""
        agent = ConcreteAgent("leader")
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        normalized = agent._normalize_observation(obs)
        
        assert normalized.shape == (1, Constants.LEADER_MESSAGE_SIZE)
        np.testing.assert_array_equal(normalized[0], obs)
    
    def test_normalize_observation_too_small(self):
        """Test observation normalization with smaller input."""
        agent = ConcreteAgent("leader")
        obs = np.array([1, 2, 3])
        
        normalized = agent._normalize_observation(obs)
        
        assert normalized.shape == (1, Constants.LEADER_MESSAGE_SIZE)
        expected = np.array([1, 2, 3, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(normalized[0], expected)
    
    def test_normalize_observation_too_large(self):
        """Test observation normalization with larger input."""
        agent = ConcreteAgent("leader")
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        normalized = agent._normalize_observation(obs)
        
        assert normalized.shape == (1, Constants.LEADER_MESSAGE_SIZE)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(normalized[0], expected)
    
    def test_get_action_from_predictions_numpy(self):
        """Test converting numpy predictions to action."""
        agent = ConcreteAgent("leader")
        predictions = np.array([[0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        action = agent._get_action_from_predictions(predictions)
        
        assert action == ActionSpace.LEFT.value  # Index 2 has highest probability
    
    def test_get_action_from_predictions_tensor(self):
        """Test converting tensor predictions to action."""
        agent = ConcreteAgent("leader")
        
        # Mock tensor with numpy method
        mock_tensor = Mock()
        mock_tensor.numpy.return_value = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        action = agent._get_action_from_predictions(mock_tensor)
        
        assert action == ActionSpace.DOWN.value  # Index 1 has highest probability
    
    def test_repr(self):
        """Test string representation of agent."""
        agent = ConcreteAgent("leader")
        agent.set_position((1, 2))
        
        repr_str = repr(agent)
        
        assert "ConcreteAgent" in repr_str
        assert f"id={agent.agent_id}" in repr_str
        assert "role=leader" in repr_str
        assert "position=(1, 2)" in repr_str