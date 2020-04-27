import tensorflow as tf

class EMA:
  def __init__(self, decay, size, value=None, dtype=tf.float32):
    self.dtype = dtype
    self.size = size
    if value is None:
      self.value = tf.ones(self.size, dtype=self.dtype)
    else:
      self.value = value
    self.decay = tf.constant(decay, dtype=self.dtype)
    self.one_minus_decay = tf.constant(1 - decay, dtype=self.dtype)
  
  def update(self, value):
    if self.value is None:
        self.value = tf.reduce_mean(tf.abs(value), axis=0, keepdims=True)
    else:
        self.value = tf.reduce_mean(tf.abs(value), axis=0, keepdims=True) * self.one_minus_decay + self.value * self.decay
    return self.value
  
  def get(self):
    return self.value