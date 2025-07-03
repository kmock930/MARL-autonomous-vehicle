"""
Critic network for value estimation in MARL.
"""

from typing import Optional
import numpy as np

from ..utils import Constants

# Conditional TensorFlow import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create mock tf for testing
    class tf:
        class keras:
            class Model:
                def predict(self, inputs, **kwargs):
                    return np.random.random((1, 1))
            class layers:
                @staticmethod
                def Input(**kwargs):
                    return None
                @staticmethod
                def Dense(**kwargs):
                    return None
            class models:
                @staticmethod
                def Model(*args, **kwargs):
                    return tf.keras.Model()
            class optimizers:
                @staticmethod
                def Adam(**kwargs):
                    return None


class CriticNetwork:
    """
    Critic network for value function estimation.
    """
    
    def __init__(
        self,
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64,
        output_size: int = 1
    ):
        """
        Initialize critic network.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layers
            output_size: Size of output layer (typically 1 for value)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the critic network model."""
        if not TF_AVAILABLE:
            return tf.keras.Model()
        
        input_layer = tf.keras.layers.Input(shape=(self.input_size,))
        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(input_layer)
        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)
        output_layer = tf.keras.layers.Dense(self.output_size, activation="linear")(x)
        
        model = tf.keras.models.Model(input_layer, output_layer, name="critic")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Constants.DEFAULT_LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def predict(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict value for given state.
        
        Args:
            state: Input state
            **kwargs: Additional arguments for prediction
            
        Returns:
            Predicted value
        """
        if not TF_AVAILABLE:
            return np.random.random((1, 1))
        return self.model.predict(state, **kwargs)
    
    def train_on_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Train the critic on a batch of data.
        
        Args:
            states: Batch of states
            targets: Target values
            
        Returns:
            Training loss
        """
        if not TF_AVAILABLE:
            return 0.0
        return self.model.train_on_batch(states, targets)
    
    def save(self, filepath: str) -> None:
        """Save the model to file."""
        if TF_AVAILABLE:
            self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """Load the model from file."""
        if TF_AVAILABLE:
            self.model = tf.keras.models.load_model(filepath)
    
    def get_weights(self):
        """Get model weights."""
        if TF_AVAILABLE:
            return self.model.get_weights()
        return []
    
    def set_weights(self, weights):
        """Set model weights."""
        if TF_AVAILABLE:
            self.model.set_weights(weights)
    
    def __call__(self, *args, **kwargs):
        """Make the network callable."""
        return self.model(*args, **kwargs)