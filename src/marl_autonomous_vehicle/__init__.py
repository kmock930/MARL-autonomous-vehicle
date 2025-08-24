"""
MARL Autonomous Vehicle Package

A modular, testable Multi-Agent Reinforcement Learning system for autonomous vehicle coordination.
This package provides leader-follower dynamics with encoder-decoder communication.
"""

# Import core components for easy access
try:
    from .agents import LeaderAgent, FollowerAgent
    from .models import PolicyNetwork, EncoderDecoder
    from .training import MAPPOTrainer, TrainingConfig
    from .utils import ACTION_SPACE, LEADER_MESSAGE_SIZE
    from .environment import SimpleGridWrapper
except ImportError:
    # Graceful degradation if dependencies are missing
    pass

__version__ = "1.0.0"
__author__ = "MARL Team"

# Export main classes for backward compatibility
__all__ = [
    "LeaderAgent", 
    "FollowerAgent", 
    "PolicyNetwork", 
    "EncoderDecoder",
    "MAPPOTrainer", 
    "TrainingConfig",
    "ACTION_SPACE", 
    "LEADER_MESSAGE_SIZE",
    "SimpleGridWrapper"
]