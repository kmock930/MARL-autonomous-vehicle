#!/usr/bin/env python3
"""
Example script demonstrating the new modular MARL system.

This script shows how to use the refactored MARL autonomous vehicle system
for both basic agent interactions and full training workflows.
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from marl_autonomous_vehicle import (
    LeaderAgent, 
    FollowerAgent, 
    MAPPOTrainer, 
    TrainingConfig, 
    SimpleGridWrapper,
    ACTION_SPACE,
    LEADER_MESSAGE_SIZE
)


def basic_agent_example():
    """Demonstrate basic agent functionality without ML dependencies."""
    print("🚀 Basic Agent Example (No ML dependencies required)")
    print("=" * 60)
    
    # Create agents
    leader = LeaderAgent()
    follower = FollowerAgent()
    
    print(f"Created agents - Leader ID: {leader.agent_id}, Follower ID: {follower.agent_id}")
    
    # Create observation
    observation = [1, 2, 3, 4, 5, 6, 7, 8]
    print(f"Observation: {observation}")
    
    # Leader acts and generates message
    leader_action = leader.act(observation)
    message = leader.get_message()
    
    print(f"Leader action: {leader_action}")
    print(f"Leader message shape: {message.shape}")
    
    # Follower acts based on observation and leader message
    follower_action = follower.act(observation, message)
    
    print(f"Follower action: {follower_action}")
    
    # Show action space
    print(f"\nAvailable actions ({len(ACTION_SPACE)} total):")
    for action in ACTION_SPACE:
        print(f"  {action.name}: {action.value}")
    
    print(f"\nLeader message size: {LEADER_MESSAGE_SIZE}")
    print("✅ Basic agent functionality working!\n")


def environment_example():
    """Demonstrate environment interaction."""
    print("🌍 Environment Interaction Example")
    print("=" * 60)
    
    # Create environment
    env = SimpleGridWrapper(
        row_size=5, 
        col_size=5, 
        num_robots=2, 
        num_leaders=1
    )
    
    # Create agents
    leader = LeaderAgent()
    follower = FollowerAgent()
    
    print("Created environment and agents")
    
    # Reset environment
    observations = env.reset()
    print(f"Initial observations: {list(observations.keys())}")
    
    # Run a few steps
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Get observation for each agent
        leader_obs = list(observations.values())[0]
        follower_obs = list(observations.values())[1] if len(observations) > 1 else leader_obs
        
        # Agents act
        leader_action = leader.act(leader_obs)
        message = leader.get_message()
        follower_action = follower.act(follower_obs, message)
        
        print(f"Leader action: {leader_action}")
        print(f"Follower action: {follower_action}")
        
        # Environment step
        actions = {0: leader_action, 1: follower_action}
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        print(f"Rewards: {rewards}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print("Episode ended")
            break
    
    print("✅ Environment interaction working!\n")


def training_example():
    """Demonstrate training workflow."""
    print("🎯 Training Example (Mock training without ML dependencies)")
    print("=" * 60)
    
    # Create configuration
    config = TrainingConfig(
        episodes=5,  # Short training for demo
        learning_rate=0.001,
        hidden_units=64
    )
    
    print(f"Training configuration: {config.episodes} episodes")
    
    # Create environment and agents
    env = SimpleGridWrapper(row_size=8, col_size=8)
    leader = LeaderAgent()
    follower = FollowerAgent()
    
    # Create trainer
    trainer = MAPPOTrainer(
        config=config,
        leader_agent=leader,
        follower_agent=follower,
        environment=env
    )
    
    print("Created MAPPO trainer")
    
    # Run training
    print("\nStarting training...")
    history = trainer.train()
    
    # Show results
    print(f"\nTraining completed!")
    print(f"Final metrics:")
    print(f"  Policy Loss: {history['policy_loss'][-1]:.4f}")
    print(f"  Value Loss: {history['value_loss'][-1]:.4f}")
    print(f"  Reward: {history['rewards'][-1]:.2f}")
    print(f"  Episode Length: {history['episode_lengths'][-1]}")
    
    # Show improvement over time
    initial_reward = history['rewards'][0]
    final_reward = history['rewards'][-1]
    improvement = final_reward - initial_reward
    
    print(f"\nReward improvement: {improvement:.2f} ({improvement/abs(initial_reward)*100:.1f}%)")
    print("✅ Training workflow working!\n")


def configuration_example():
    """Demonstrate configuration management."""
    print("⚙️  Configuration Management Example")
    print("=" * 60)
    
    # Create default config
    config = TrainingConfig()
    print(f"Default config: {config.episodes} episodes, {config.learning_rate} learning rate")
    
    # Update configuration
    new_config = config.update(episodes=2000, learning_rate=0.01, batch_size=64)
    print(f"Updated config: {new_config.episodes} episodes, {new_config.learning_rate} learning rate")
    print(f"Original unchanged: {config.episodes} episodes")
    
    # Serialize configuration
    config_dict = config.to_dict()
    print(f"Serialized config has {len(config_dict)} parameters")
    
    # Deserialize configuration
    restored_config = TrainingConfig.from_dict(config_dict)
    print(f"Restored config: {restored_config.episodes} episodes")
    
    print("✅ Configuration management working!\n")


def main():
    """Run all examples."""
    print("🎉 MARL Autonomous Vehicle - Modular System Demo")
    print("=" * 60)
    print("This demo shows the refactored modular MARL system in action.")
    print("All examples work without TensorFlow or other ML dependencies!\n")
    
    try:
        basic_agent_example()
        environment_example()
        training_example()
        configuration_example()
        
        print("🎊 All examples completed successfully!")
        print("\nThe modular MARL system is working perfectly!")
        print("You can now:")
        print("  - Use agents in your own projects")
        print("  - Extend the system with new components")
        print("  - Run full training with TensorFlow")
        print("  - Deploy in production environments")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())