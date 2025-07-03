"""
Policy network models for agent decision making.
"""

from typing import Optional
import numpy as np

from ..utils import ActionSpace, Constants

# Conditional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create mock classes for testing without TensorFlow
    class tf:
        class keras:
            class Model:
                def __init__(self, *args, **kwargs):
                    pass
                def call(self, inputs):
                    return np.random.random((1, len(ActionSpace)))
                def predict(self, inputs, **kwargs):
                    return np.random.random((1, len(ActionSpace)))
            class layers:
                @staticmethod
                def Input(**kwargs):
                    return None
                @staticmethod
                def Dense(**kwargs):
                    return None
                @staticmethod
                def GRU(**kwargs):
                    return None
                @staticmethod
                def Reshape(**kwargs):
                    return None
            class models:
                @staticmethod
                def Model(*args, **kwargs):
                    return tf.keras.Model()


class PolicyNetwork:
    """
    Factory class for creating policy networks.
    """
    
    @staticmethod
    def create_leader_policy(
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64,
        output_size: int = len(ActionSpace)
    ):
        """
        Create a leader policy network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layers
            output_size: Size of output layer (number of actions)
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            return tf.keras.Model()
        
        input_layer = tf.keras.layers.Input(shape=(input_size,))
        x = tf.keras.layers.Dense(hidden_size, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(hidden_size, activation="relu")(x)
        output_layer = tf.keras.layers.Dense(output_size, activation="softmax")(x)
        
        model = tf.keras.models.Model(input_layer, output_layer, name="leader_policy")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Constants.DEFAULT_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @staticmethod
    def create_follower_policy(
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64,
        output_size: int = len(ActionSpace)
    ):
        """
        Create a follower policy network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layers
            output_size: Size of output layer (number of actions)
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            return tf.keras.Model()
        
        input_layer = tf.keras.layers.Input(shape=(input_size,))
        x = tf.keras.layers.Dense(hidden_size, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(hidden_size, activation="relu")(x)
        output_layer = tf.keras.layers.Dense(output_size, activation="softmax")(x)
        
        model = tf.keras.models.Model(input_layer, output_layer, name="follower_policy")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Constants.DEFAULT_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class LeaderPolicyNetwork:
    """Leader-specific policy network implementation."""
    
    def __init__(
        self,
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64,
        output_size: int = len(ActionSpace)
    ):
        """
        Initialize leader policy network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layers
            output_size: Size of output layer
        """
        self.model = PolicyNetwork.create_leader_policy(input_size, hidden_size, output_size)
    
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the policy network."""
        if not TF_AVAILABLE:
            return np.random.random((1, len(ActionSpace)))
        return self.model.predict(observation, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class FollowerPolicyNetwork:
    """Follower-specific policy network implementation."""
    
    def __init__(
        self,
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64,
        output_size: int = len(ActionSpace)
    ):
        """
        Initialize follower policy network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layers  
            output_size: Size of output layer
        """
        self.model = PolicyNetwork.create_follower_policy(input_size, hidden_size, output_size)
    
    def predict(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions using the policy network."""
        if not TF_AVAILABLE:
            return np.random.random((1, len(ActionSpace)))
        return self.model.predict(observation, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)