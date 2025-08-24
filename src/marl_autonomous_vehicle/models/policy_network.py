"""
Policy network implementations for leader and follower agents.
"""

from typing import Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    TF_AVAILABLE = False

from ..utils.constants import LEADER_MESSAGE_SIZE


class PolicyNetwork:
    """
    Factory class for creating policy networks for different agent types.
    Provides both leader and follower policy architectures.
    """
    
    @staticmethod
    def create_leader_policy(input_shape: Tuple[int, ...] = (LEADER_MESSAGE_SIZE,),
                           num_actions: int = 9,
                           hidden_units: int = 64) -> Optional['keras.Model']:
        """
        Create a policy network for leader agents.
        
        Args:
            input_shape: Input shape for observations
            num_actions: Number of possible actions
            hidden_units: Number of hidden units in dense layers
            
        Returns:
            Compiled Keras model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            keras.layers.Dense(hidden_units, activation='relu', input_shape=input_shape),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(num_actions, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_follower_policy(input_shape: Tuple[int, ...] = (2, LEADER_MESSAGE_SIZE),
                             num_actions: int = 9,
                             hidden_units: int = 64) -> Optional['keras.Model']:
        """
        Create a policy network for follower agents.
        Handles combined observation and message input.
        
        Args:
            input_shape: Input shape for combined observation and message
            num_actions: Number of possible actions
            hidden_units: Number of hidden units in dense layers
            
        Returns:
            Compiled Keras model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(num_actions, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_critic_network(input_shape: Tuple[int, ...] = (LEADER_MESSAGE_SIZE,),
                            hidden_units: int = 64) -> Optional['keras.Model']:
        """
        Create a critic network for value estimation.
        
        Args:
            input_shape: Input shape for state observations
            hidden_units: Number of hidden units in dense layers
            
        Returns:
            Compiled Keras model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            keras.layers.Dense(hidden_units, activation='relu', input_shape=input_shape),
            keras.layers.Dense(hidden_units, activation='relu'),
            keras.layers.Dense(1)  # Single value output
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model