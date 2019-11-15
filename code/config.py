import tensorflow as tf

tf.flags.DEFINE_integer('batch_size', 32, '')
tf.flags.DEFINE_string('name', 'exp', '')
tf.flags.DEFINE_integer('restore_epoch', -1, '')
tf.flags.DEFINE_integer('restore_batch', -1, '')
tf.flags.DEFINE_integer('epochs', 100, '')
tf.flags.DEFINE_integer('langevin_steps', 90, '')
tf.flags.DEFINE_float('step_size', 0.1, '')
tf.flags.DEFINE_bool('situation_invariant', False, '')
tf.flags.DEFINE_bool('adaptive_langevin', False, '')
tf.flags.DEFINE_bool('clip_norm_langevin', False, '')
tf.flags.DEFINE_integer('two_stage_optim', -1, '')
tf.flags.DEFINE_bool('debug', False, '')
tf.flags.DEFINE_float('d_lr', 1e-3, '')
tf.flags.DEFINE_float('beta1', 0.99, '')
tf.flags.DEFINE_float('beta2', 0.999, '')
tf.flags.DEFINE_boolean('tb_render', False, '')

flags = tf.flags.FLAGS