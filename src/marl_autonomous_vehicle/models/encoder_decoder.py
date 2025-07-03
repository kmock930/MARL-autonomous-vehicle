"""
Encoder-decoder networks for agent communication.
"""

from typing import Optional, Tuple
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
                    return np.random.random((1, Constants.LEADER_MESSAGE_SIZE))
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
            class optimizers:
                @staticmethod
                def Adam(**kwargs):
                    return None


class Encoder:
    """
    Encoder network for compressing leader messages.
    """
    
    def __init__(
        self,
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        latent_size: int = Constants.LEADER_MESSAGE_SIZE,
        hidden_size: int = 64
    ):
        """
        Initialize encoder network.
        
        Args:
            input_size: Size of input layer
            latent_size: Size of latent/encoded representation
            hidden_size: Size of hidden layers
        """
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the encoder network model."""
        if not TF_AVAILABLE:
            return tf.keras.Model()
        
        input_layer = tf.keras.layers.Input(shape=(self.input_size,))
        reshaped = tf.keras.layers.Reshape((1, self.input_size))(input_layer)
        x = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)(reshaped)
        latent = tf.keras.layers.GRU(self.latent_size)(x)
        
        model = tf.keras.models.Model(input_layer, latent, name="encoder")
        return model
    
    def encode(self, message: np.ndarray, **kwargs) -> np.ndarray:
        """
        Encode a message.
        
        Args:
            message: Input message to encode
            **kwargs: Additional arguments for prediction
            
        Returns:
            Encoded message
        """
        if not TF_AVAILABLE:
            return np.random.random((1, self.latent_size))
        return self.model.predict(message, **kwargs)
    
    def predict(self, message: np.ndarray, **kwargs) -> np.ndarray:
        """Alias for encode method."""
        return self.encode(message, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the encoder callable."""
        return self.model(*args, **kwargs)


class Decoder:
    """
    Decoder network for reconstructing messages from encoded representations.
    """
    
    def __init__(
        self,
        latent_size: int = Constants.LEADER_MESSAGE_SIZE,
        output_size: int = 32,  # Larger decoded message
        hidden_size: int = 64
    ):
        """
        Initialize decoder network.
        
        Args:
            latent_size: Size of latent/encoded input
            output_size: Size of decoded output
            hidden_size: Size of hidden layers
        """
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the decoder network model."""
        if not TF_AVAILABLE:
            return tf.keras.Model()
        
        latent_input = tf.keras.layers.Input(shape=(self.latent_size,))
        reshaped = tf.keras.layers.Reshape((1, self.latent_size))(latent_input)
        x = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)(reshaped)
        x = tf.keras.layers.GRU(self.hidden_size)(x)
        output_layer = tf.keras.layers.Dense(self.output_size, activation="linear")(x)
        
        model = tf.keras.models.Model(latent_input, output_layer, name="decoder")
        return model
    
    def decode(self, encoded_message: np.ndarray, **kwargs) -> np.ndarray:
        """
        Decode an encoded message.
        
        Args:
            encoded_message: Encoded message to decode
            **kwargs: Additional arguments for prediction
            
        Returns:
            Decoded message
        """
        if not TF_AVAILABLE:
            return np.random.random((1, self.output_size))
        return self.model.predict(encoded_message, **kwargs)
    
    def predict(self, encoded_message: np.ndarray, **kwargs) -> np.ndarray:
        """Alias for decode method."""
        return self.decode(encoded_message, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the decoder callable."""
        return self.model(*args, **kwargs)


class EncoderDecoder:
    """
    Combined encoder-decoder system for agent communication.
    """
    
    def __init__(
        self,
        input_size: int = Constants.LEADER_MESSAGE_SIZE,
        latent_size: int = Constants.LEADER_MESSAGE_SIZE,
        output_size: int = 32,
        hidden_size: int = 64
    ):
        """
        Initialize encoder-decoder system.
        
        Args:
            input_size: Size of input messages
            latent_size: Size of encoded representation
            output_size: Size of decoded messages
            hidden_size: Size of hidden layers
        """
        self.encoder = Encoder(input_size, latent_size, hidden_size)
        self.decoder = Decoder(latent_size, output_size, hidden_size)
    
    def encode_decode(self, message: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a message and then decode it (for training/testing).
        
        Args:
            message: Input message
            **kwargs: Additional arguments for prediction
            
        Returns:
            Tuple of (encoded_message, decoded_message)
        """
        encoded = self.encoder.encode(message, **kwargs)
        decoded = self.decoder.decode(encoded, **kwargs)
        return encoded, decoded
    
    def get_encoder(self) -> Encoder:
        """Get the encoder component."""
        return self.encoder
    
    def get_decoder(self) -> Decoder:
        """Get the decoder component."""
        return self.decoder


def contrastive_loss(messages: np.ndarray, positive_pairs: np.ndarray, temperature: float = 0.1) -> float:
    """
    Compute contrastive loss for message learning.
    
    Args:
        messages: Batch of messages
        positive_pairs: Pairs of positive examples
        temperature: Temperature parameter for contrastive loss
        
    Returns:
        Contrastive loss value
    """
    if not TF_AVAILABLE:
        return 0.0
    
    # This is a simplified implementation
    # In practice, you would implement proper contrastive loss
    similarity_matrix = tf.linalg.matmul(messages, messages, transpose_b=True)
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels for positive pairs
    labels = tf.eye(tf.shape(messages)[0])
    
    # Compute cross-entropy loss
    loss = tf.keras.losses.categorical_crossentropy(labels, similarity_matrix)
    return tf.reduce_mean(loss)