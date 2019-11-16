import tensorflow as tf

tf.flags.DEFINE_string('name', 'exp', '')
tf.flags.DEFINE_integer('cup_id', 1, '')
tf.flags.DEFINE_integer('batch_size', 1024, '')

tf.flags.DEFINE_integer('epochs', 200, '')
tf.flags.DEFINE_float('d_lr', 1e-5, '')
tf.flags.DEFINE_float('beta1', 0.99, '')
tf.flags.DEFINE_float('beta2', 0.999, '')
tf.flags.DEFINE_float('delta', 0.1, '')

config = tf.flags.FLAGS