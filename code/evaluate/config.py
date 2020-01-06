import tensorflow as tf

# Meta
tf.flags.DEFINE_string('name', 'static_z2_physical_prior_nobn', '')
tf.flags.DEFINE_integer('restore_epoch', 35, '')
tf.flags.DEFINE_integer('restore_batch', 0, '')
tf.flags.DEFINE_string('restore_name', '', '')
tf.flags.DEFINE_bool('debug', False, '')
tf.flags.DEFINE_boolean('tb_render', False, '')

# Model
tf.flags.DEFINE_boolean('dynamic_z2', False, '')
tf.flags.DEFINE_integer('z2_size', 10, '')
tf.flags.DEFINE_float('random_scale', 0.3, 'for initializing syn position')
tf.flags.DEFINE_float('prior_weight', 10, '')
tf.flags.DEFINE_float('random_strength', 0.0, 'for langevin dynamics')
tf.flags.DEFINE_string('prior_type', 'Phys', 'choose between "NN" and "Phys"')

# Train
tf.flags.DEFINE_integer('epochs', 100, '')
tf.flags.DEFINE_integer('batch_size', 24, '')
tf.flags.DEFINE_float('d_lr', 1e-3, '')
tf.flags.DEFINE_float('beta1', 0.99, '')
tf.flags.DEFINE_float('beta2', 0.999, '')
tf.flags.DEFINE_float('l2_reg', 0.001, '')

# Langevin
tf.flags.DEFINE_float('step_size', 0.1, '')
tf.flags.DEFINE_integer('langevin_steps', 90, '')
tf.flags.DEFINE_bool('adaptive_langevin', True, '')
tf.flags.DEFINE_bool('clip_norm_langevin', True, '')

# Touch Filter
tf.flags.DEFINE_bool('situation_invariant', False, '')


flags = tf.flags.FLAGS