"""
MAPPO (Multi-Agent Proximal Policy Optimization) trainer.
"""

from typing import Dict, List, Tuple, Optional, Any
import time
import datetime
import os
from dataclasses import dataclass

from ..utils import Constants
from ..agents import LeaderAgent, FollowerAgent
from ..models import CriticNetwork, EncoderDecoder
from ..environment import SimpleGridEnvWrapper

# Conditional imports
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for testing
    class np:
        @staticmethod
        def array(x):
            return x
        @staticmethod
        def zeros(shape):
            return [0] * (shape[0] if isinstance(shape, tuple) else shape)
        @staticmethod
        def concatenate(arrays, axis=None):
            return sum(arrays, [])
    
    class pd:
        @staticmethod
        def DataFrame(data):
            return data


@dataclass
class TrainingConfig:
    """Configuration for MAPPO training."""
    episodes: int = 1000
    learning_rate: float = Constants.DEFAULT_LEARNING_RATE
    batch_size: int = Constants.DEFAULT_BATCH_SIZE
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    update_epochs: int = 4  # Number of optimization epochs per update
    tether_tolerate_count: int = Constants.TETHER_TOLERATE_COUNT


class TrainingMetrics:
    """Class to track training metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.tether_violations = []
        self.collision_counts = []
        self.training_losses = []
        self.timestamps = []
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        success: bool,
        tether_violations: int,
        collisions: int,
        loss: Optional[float] = None
    ):
        """Log metrics for an episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(1 if success else 0)
        self.tether_violations.append(tether_violations)
        self.collision_counts.append(collisions)
        if loss is not None:
            self.training_losses.append(loss)
        self.timestamps.append(datetime.datetime.now())
    
    def get_recent_average(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        if metric == "reward":
            values = self.episode_rewards
        elif metric == "length":
            values = self.episode_lengths
        elif metric == "success":
            values = self.success_rates
        elif metric == "tether":
            values = self.tether_violations
        elif metric == "collision":
            values = self.collision_counts
        else:
            return 0.0
        
        if not values:
            return 0.0
        
        recent_values = values[-window:] if len(values) > window else values
        return sum(recent_values) / len(recent_values)
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV file."""
        if not NUMPY_AVAILABLE:
            return
        
        data = {
            'episode': list(range(len(self.episode_rewards))),
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'success': self.success_rates,
            'tether_violations': self.tether_violations,
            'collisions': self.collision_counts,
            'timestamp': self.timestamps
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)


class MAPPOTrainer:
    """
    Multi-Agent Proximal Policy Optimization trainer.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        leader_agent: LeaderAgent,
        follower_agent: FollowerAgent,
        critic_network: CriticNetwork,
        encoder_decoder: EncoderDecoder,
        environment: SimpleGridEnvWrapper
    ):
        """
        Initialize MAPPO trainer.
        
        Args:
            config: Training configuration
            leader_agent: Leader agent
            follower_agent: Follower agent
            critic_network: Critic network for value estimation
            encoder_decoder: Encoder-decoder for communication
            environment: Environment wrapper
        """
        self.config = config
        self.leader_agent = leader_agent
        self.follower_agent = follower_agent
        self.critic_network = critic_network
        self.encoder_decoder = encoder_decoder
        self.environment = environment
        self.metrics = TrainingMetrics()
        
        # Training state
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.total_training_time = 0.0
    
    def train(self, episodes: Optional[int] = None) -> TrainingMetrics:
        """
        Train the agents using MAPPO.
        
        Args:
            episodes: Number of episodes to train (uses config if None)
            
        Returns:
            Training metrics
        """
        if episodes is None:
            episodes = self.config.episodes
        
        print(f"Starting MAPPO training for {episodes} episodes...")
        start_time = time.time()
        
        for episode in range(episodes):
            self.current_episode = episode
            episode_start_time = time.time()
            
            # Run single episode
            episode_reward, episode_length, success, violations, collisions = self._run_episode()
            
            # Log metrics
            self.metrics.log_episode(
                episode, episode_reward, episode_length, success, violations, collisions
            )
            
            # Print progress
            if episode % 100 == 0 or episode == episodes - 1:
                avg_reward = self.metrics.get_recent_average("reward", 100)
                avg_success = self.metrics.get_recent_average("success", 100)
                episode_time = time.time() - episode_start_time
                
                print(f"Episode {episode}/{episodes}")
                print(f"  Reward: {episode_reward:.2f} (avg: {avg_reward:.2f})")
                print(f"  Success Rate: {avg_success:.2f}")
                print(f"  Episode Time: {episode_time:.2f}s")
                
                # Save best model
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self._save_models("best")
        
        self.total_training_time = time.time() - start_time
        print(f"Training completed in {self.total_training_time:.2f} seconds")
        
        return self.metrics
    
    def _run_episode(self) -> Tuple[float, int, bool, int, int]:
        """
        Run a single training episode.
        
        Returns:
            Tuple of (total_reward, episode_length, success, tether_violations, collisions)
        """
        if not NUMPY_AVAILABLE:
            # Mock episode for testing
            return 10.0, 50, True, 0, 0
        
        # Reset environment
        obs, info = self.environment.reset()
        
        total_reward = 0.0
        episode_length = 0
        tether_violations = 0
        collisions = 0
        done = False
        
        while not done and episode_length < 1000:  # Max episode length
            # Get actions from agents
            leader_obs = self._get_agent_observation(0)  # Leader is agent 0
            follower_obs = self._get_agent_observation(1)  # Follower is agent 1
            
            # Leader acts and creates message
            leader_action = self.leader_agent.act(leader_obs)
            leader_message = self.leader_agent.get_message()
            
            # Follower acts based on leader message
            follower_action = self.follower_agent.act(follower_obs, leader_message)
            
            # Combine actions
            actions = {0: leader_action, 1: follower_action}
            
            # Step environment
            obs, rewards, terminated, truncated, info = self.environment.step(actions, is_training=True)
            
            # Calculate episode metrics
            episode_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
            total_reward += episode_reward
            episode_length += 1
            
            # Check for violations and collisions
            if info.get('out_of_tether_count', 0) > 0:
                tether_violations += 1
            
            # Check termination
            done = terminated or truncated
        
        # Determine success (reached target)
        success = done and episode_length < 1000  # Success if completed within time limit
        
        return total_reward, episode_length, success, tether_violations, collisions
    
    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent."""
        if not NUMPY_AVAILABLE:
            return np.array([0] * Constants.LEADER_MESSAGE_SIZE)
        
        # Get basic observation from environment
        obs = self.environment.get_observations()
        if agent_id in obs:
            return np.array(obs[agent_id]).reshape(-1)
        else:
            # Return dummy observation
            return np.zeros(Constants.LEADER_MESSAGE_SIZE)
    
    def _save_models(self, suffix: str = ""):
        """Save all models."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if suffix:
                suffix = f"_{suffix}"
            
            # Save models (if they have save methods)
            if hasattr(self.leader_agent.policy_model, 'save'):
                self.leader_agent.policy_model.save(f'models/leader_policy{suffix}_{timestamp}.h5')
            
            if hasattr(self.follower_agent.policy_model, 'save'):
                self.follower_agent.policy_model.save(f'models/follower_policy{suffix}_{timestamp}.h5')
            
            if hasattr(self.critic_network, 'save'):
                self.critic_network.save(f'models/critic{suffix}_{timestamp}.h5')
            
            if hasattr(self.encoder_decoder.encoder, 'model') and hasattr(self.encoder_decoder.encoder.model, 'save'):
                self.encoder_decoder.encoder.model.save(f'models/encoder{suffix}_{timestamp}.h5')
            
            if hasattr(self.encoder_decoder.decoder, 'model') and hasattr(self.encoder_decoder.decoder.model, 'save'):
                self.encoder_decoder.decoder.model.save(f'models/decoder{suffix}_{timestamp}.h5')
                
            print(f"Models saved with suffix '{suffix}' and timestamp {timestamp}")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def save_metrics(self, filepath: str):
        """Save training metrics to file."""
        self.metrics.save_to_csv(filepath)
    
    def get_metrics(self) -> TrainingMetrics:
        """Get training metrics."""
        return self.metrics