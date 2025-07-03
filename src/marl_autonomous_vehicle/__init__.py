"""
MARL Autonomous Vehicle Package

A Multi-Agent Reinforcement Learning system for autonomous vehicle coordination
using leader-follower dynamics in a grid-based environment.
"""

__version__ = "1.0.0"
__author__ = "MARL Autonomous Vehicle Team"

# Import basic utilities that don't require external dependencies
from .utils import ActionSpace, Rewards, Constants, GridElements, AgentRoles

# Conditional imports for components that may require external dependencies
try:
    from .environment import SimpleGridEnvWrapper
    from .agents import Agent, LeaderAgent, FollowerAgent
    from .models import PolicyNetwork, CriticNetwork, EncoderDecoder
    from .training import MAPPOTrainer
    
    __all__ = [
        "SimpleGridEnvWrapper",
        "Agent",
        "LeaderAgent", 
        "FollowerAgent",
        "PolicyNetwork",
        "CriticNetwork",
        "EncoderDecoder", 
        "MAPPOTrainer",
        "ActionSpace",
        "Rewards",
        "Constants",
        "GridElements",
        "AgentRoles"
    ]
except ImportError as e:
    # If dependencies are missing, only export utils
    __all__ = [
        "ActionSpace",
        "Rewards", 
        "Constants",
        "GridElements",
        "AgentRoles"
    ]