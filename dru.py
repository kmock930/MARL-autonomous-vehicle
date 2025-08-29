import numpy as np
import tensorflow as tf

# https://github.com/minqi/learning-to-communicate-pytorch/
class DRU_DIAL(tf.keras.Model):
	def __init__(self, sigma, comm_narrow=True, hard=False):
		super(DRU, self).__init__()
		self.sigma = sigma
		self.comm_narrow = comm_narrow
		self.hard = hard

	def regularize(self, m):
		m_reg = m + tf.random.normal(tf.shape(m)) * self.sigma
		if self.comm_narrow:
			m_reg = tf.sigmoid(m_reg)
		else:
			m_reg = tf.nn.softmax(m_reg, axis=0)
		return m_reg

	def discretize(self, m):
		if self.hard:
			if self.comm_narrow:
				return tf.cast(tf.math.sign(tf.cast(m > 0.5, tf.float32) - 0.5), tf.float32)
			else:
				m_ = tf.zeros_like(m)
				if len(m.shape) == 1:
					idx = tf.argmax(m, axis=0)
					m_ = tf.tensor_scatter_nd_update(m_, [[idx]], [1.])
				elif len(m.shape) == 2:
					idx = tf.argmax(m, axis=1)
					m_ = tf.tensor_scatter_nd_update(
						m_,
						tf.stack([tf.range(tf.shape(m)[0]), idx], axis=1),
						tf.ones(tf.shape(m)[0])
					)
				else:
					raise ValueError('Wrong message shape: {}'.format(m.shape))
				return m_
		else:
			scale = 2 * 20
			if self.comm_narrow:
				return tf.sigmoid((tf.cast(m > 0.5, tf.float32) - 0.5) * scale)
			else:
				return tf.nn.softmax(m * scale, axis=-1)

	def call(self, m, train_mode):
		if train_mode:
			return self.regularize(m)
		else:
			return self.discretize(m)
