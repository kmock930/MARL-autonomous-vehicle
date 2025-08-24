"""
Encoder-decoder networks for agent communication.
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


class EncoderDecoder:
    """
    Factory class for creating encoder-decoder networks for agent communication.
    Handles message encoding by leaders and decoding by followers.
    """
    
    @staticmethod
    def create_encoder(input_shape: Tuple[int, ...] = (LEADER_MESSAGE_SIZE,),
                      encoded_dim: int = 32,
                      hidden_units: int = 64) -> Optional['keras.Model']:
        """
        Create an encoder network for compressing leader messages.
        
        Args:
            input_shape: Input shape for leader messages
            encoded_dim: Dimensionality of encoded representation
            hidden_units: Number of hidden units in GRU layers
            
        Returns:
            Compiled Keras encoder model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Reshape((1, input_shape[0])),
            keras.layers.GRU(hidden_units, return_sequences=True),
            keras.layers.GRU(encoded_dim, return_sequences=False)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_decoder(encoded_dim: int = 32,
                      output_shape: Tuple[int, ...] = (LEADER_MESSAGE_SIZE,),
                      hidden_units: int = 64) -> Optional['keras.Model']:
        """
        Create a decoder network for reconstructing messages from encoded representations.
        
        Args:
            encoded_dim: Dimensionality of encoded input
            output_shape: Shape of decoded output
            hidden_units: Number of hidden units in GRU layers
            
        Returns:
            Compiled Keras decoder model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None
            
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(encoded_dim,)),
            keras.layers.Reshape((1, encoded_dim)),
            keras.layers.GRU(hidden_units, return_sequences=True),
            keras.layers.GRU(output_shape[0], return_sequences=False),
            keras.layers.Dense(output_shape[0])
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_encoder_decoder_pair(input_shape: Tuple[int, ...] = (LEADER_MESSAGE_SIZE,),
                                   encoded_dim: int = 32,
                                   hidden_units: int = 64) -> Tuple[Optional['keras.Model'], Optional['keras.Model']]:
        """
        Create a matching encoder-decoder pair for communication.
        
        Args:
            input_shape: Input shape for messages
            encoded_dim: Dimensionality of encoded representation
            hidden_units: Number of hidden units in networks
            
        Returns:
            Tuple of (encoder, decoder) models or (None, None) if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            return None, None
            
        encoder = EncoderDecoder.create_encoder(input_shape, encoded_dim, hidden_units)
        decoder = EncoderDecoder.create_decoder(encoded_dim, input_shape, hidden_units)
        
        return encoder, decoder