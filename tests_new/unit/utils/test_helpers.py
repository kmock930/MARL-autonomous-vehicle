"""
Unit tests for helper functions.
"""

import pytest
import numpy as np
from src.marl_autonomous_vehicle.utils.helpers import (
    generate_map, 
    calculate_distance, 
    is_valid_position,
    get_valid_actions,
    normalize_observation
)
from src.marl_autonomous_vehicle.utils import GridElements, ActionSpace


class TestMapGeneration:
    """Test map generation functions."""
    
    def test_generate_map_basic(self):
        """Test basic map generation."""
        grid, robots, targets = generate_map(
            row_size=5,
            col_size=5,
            num_soft_obstacles=2,
            num_hard_obstacles=1,
            num_robots=2,
            tether_dist=2,
            num_targets=1
        )
        
        assert grid.shape == (5, 5)
        assert len(robots) == 2
        assert len(targets) == 1
        assert robots[0]['role'] == 'leader'
        assert robots[1]['role'] == 'follower'
    
    def test_generate_map_too_many_objects(self):
        """Test error when too many objects for grid size."""
        with pytest.raises(ValueError):
            generate_map(
                row_size=2,
                col_size=2,
                num_soft_obstacles=5,
                num_hard_obstacles=5,
                num_robots=2,
                tether_dist=1,
                num_targets=1
            )
    
    def test_generate_map_empty(self):
        """Test generating empty map."""
        grid, robots, targets = generate_map(
            row_size=3,
            col_size=3,
            num_soft_obstacles=0,
            num_hard_obstacles=0,
            num_robots=1,
            tether_dist=1,
            num_targets=1
        )
        
        assert grid.shape == (3, 3)
        assert len(robots) == 1
        assert len(targets) == 1


class TestDistanceCalculation:
    """Test distance calculation functions."""
    
    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        dist = calculate_distance((0, 0), (0, 0))
        assert dist == 0.0
    
    def test_calculate_distance_horizontal(self):
        """Test distance calculation for horizontal movement."""
        dist = calculate_distance((0, 0), (0, 3))
        assert dist == 3.0
    
    def test_calculate_distance_vertical(self):
        """Test distance calculation for vertical movement."""
        dist = calculate_distance((0, 0), (4, 0))
        assert dist == 4.0
    
    def test_calculate_distance_diagonal(self):
        """Test distance calculation for diagonal movement."""
        dist = calculate_distance((0, 0), (3, 4))
        assert dist == 5.0


class TestPositionValidation:
    """Test position validation functions."""
    
    def test_is_valid_position_in_bounds(self):
        """Test valid position within bounds."""
        obstacles = np.zeros((5, 5))
        agents = []
        assert is_valid_position((2, 2), (5, 5), obstacles, agents) is True
    
    def test_is_valid_position_out_of_bounds(self):
        """Test invalid position out of bounds."""
        obstacles = np.zeros((5, 5))
        agents = []
        assert is_valid_position((5, 5), (5, 5), obstacles, agents) is False
        assert is_valid_position((-1, 0), (5, 5), obstacles, agents) is False
    
    def test_is_valid_position_agent_collision(self):
        """Test invalid position due to agent collision."""
        obstacles = np.zeros((5, 5))
        agents = [{'position': (2, 2)}]
        assert is_valid_position((2, 2), (5, 5), obstacles, agents) is False
    
    def test_is_valid_position_hard_obstacle(self):
        """Test invalid position due to hard obstacle."""
        obstacles = np.zeros((5, 5))
        obstacles[2, 2] = GridElements.OBSTACLE_HARD
        agents = []
        assert is_valid_position((2, 2), (5, 5), obstacles, agents) is False


class TestObservationNormalization:
    """Test observation normalization functions."""
    
    def test_normalize_observation_exact_size(self):
        """Test normalization with exact target size."""
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        normalized = normalize_observation(obs, max_size=8)
        assert normalized.shape == (1, 8)
        np.testing.assert_array_equal(normalized[0], obs)
    
    def test_normalize_observation_too_small(self):
        """Test normalization with smaller observation."""
        obs = np.array([1, 2, 3])
        normalized = normalize_observation(obs, max_size=8)
        assert normalized.shape == (1, 8)
        expected = np.array([1, 2, 3, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(normalized[0], expected)
    
    def test_normalize_observation_too_large(self):
        """Test normalization with larger observation."""
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        normalized = normalize_observation(obs, max_size=8)
        assert normalized.shape == (1, 8)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(normalized[0], expected)
    
    def test_normalize_observation_2d_input(self):
        """Test normalization with 2D input."""
        obs = np.array([[1, 2], [3, 4]])
        normalized = normalize_observation(obs, max_size=8)
        assert normalized.shape == (1, 8)
        expected = np.array([1, 2, 3, 4, 0, 0, 0, 0])
        np.testing.assert_array_equal(normalized[0], expected)