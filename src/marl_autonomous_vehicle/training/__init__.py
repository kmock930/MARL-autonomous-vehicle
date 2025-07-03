"""
Training module for MAPPO and other RL algorithms.
"""

from .mappo_trainer import MAPPOTrainer
from .training_config import TrainingConfig

__all__ = ["MAPPOTrainer", "TrainingConfig"]