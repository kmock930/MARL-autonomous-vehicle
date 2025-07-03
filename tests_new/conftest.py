"""
Configuration for pytest and coverage.
"""

import sys
import os

# Add source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock TensorFlow for tests if not available
try:
    import tensorflow
except ImportError:
    import unittest.mock as mock
    sys.modules['tensorflow'] = mock.MagicMock()
    sys.modules['tensorflow.keras'] = mock.MagicMock()
    sys.modules['tensorflow.keras.layers'] = mock.MagicMock()
    sys.modules['tensorflow.keras.models'] = mock.MagicMock()
    sys.modules['tensorflow.keras.optimizers'] = mock.MagicMock()