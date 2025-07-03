"""
Agent module for MARL autonomous vehicle system.
"""

from .base_agent import BaseAgent
from .leader_agent import LeaderAgent
from .follower_agent import FollowerAgent

# For backward compatibility
Agent = BaseAgent

__all__ = ["BaseAgent", "Agent", "LeaderAgent", "FollowerAgent"]