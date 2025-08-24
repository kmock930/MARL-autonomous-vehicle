"""
Training configuration for MARL experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    Centralizes all hyperparameters and settings.
    """
    
    # Training parameters
    episodes: int = 1000
    max_steps_per_episode: int = 200
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # MAPPO specific parameters
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Communication parameters
    contrastive_loss_weight: float = 0.1
    reconstruction_loss_weight: float = 0.5
    
    # Environment parameters
    tether_tolerate_count: int = 5
    grid_size: tuple = (10, 10)
    num_obstacles_soft: int = 10
    num_obstacles_hard: int = 5
    num_robots: int = 2
    num_leaders: int = 1
    num_targets: int = 1
    tether_distance: int = 2
    
    # Model architecture
    hidden_units: int = 64
    encoded_message_dim: int = 32
    
    # Training control
    save_frequency: int = 100
    log_frequency: int = 10
    validation_frequency: int = 50
    
    # File paths
    model_save_path: str = "models/"
    log_save_path: str = "logs/"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> 'TrainingConfig':
        """Create new config with updated parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)