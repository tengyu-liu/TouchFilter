import tensorflow as tf

tf.flags.DEFINE_string('name', 'exp', '')
tf.flags.DEFINE_integer('restore_epoch', -1, '')
tf.flags.DEFINE_integer('restore_batch', -1, '')

tf.flags.DEFINE_integer('epochs', 100, '')
tf.flags.DEFINE_integer('batch_size', 32, '')
tf.flags.DEFINE_integer('z_size', 22, '')
tf.flags.DEFINE_integer('langevin_steps', 15, '')
tf.flags.DEFINE_bool('use_generator', False, '')
tf.flags.DEFINE_bool('use_pca', False, '')
tf.flags.DEFINE_float('step_size', 0.1, '')
tf.flags.DEFINE_integer('n_channel', 1, '')
tf.flags.DEFINE_bool('sigmoid_energy', False, '')

tf.flags.DEFINE_float('d_lr', 1e-3, '')
tf.flags.DEFINE_float('g_lr', 1e-3, '')
tf.flags.DEFINE_float('beta1', 0.99, '')
tf.flags.DEFINE_float('beta2', 0.999, '')

tf.flags.DEFINE_float('des_weight_norm', 1e-3, '')
tf.flags.DEFINE_float('gen_weight_norm', 1e-3, '')
tf.flags.DEFINE_float('prior_mult', 1e-3, '')

tf.flags.DEFINE_integer('n_filter', 10, '')

flags = tf.flags.FLAGS
