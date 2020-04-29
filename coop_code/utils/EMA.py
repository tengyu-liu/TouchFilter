import tensorflow as tf

class EMA:
  def __init__(self, decay, value, dtype=tf.float32):
    self.dtype = dtype
    self.value = value
    self.decay = tf.constant(decay, dtype=self.dtype)
    self.one_minus_decay = tf.constant(1 - decay, dtype=self.dtype)
  
  def update(self, value):
    self.value *= self.decay
    self.value += tf.reduce_mean(tf.abs(value), axis=0, keepdims=True) * self.one_minus_decay
  
  def get(self):
    return self.value