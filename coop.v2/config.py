import tensorflow as tf

# Meta
tf.flags.DEFINE_string('name', 'exp', '')
tf.flags.DEFINE_integer('restore_epoch', -1, '')
tf.flags.DEFINE_string('restore_name', '', '')
tf.flags.DEFINE_bool('debug', False, '')
tf.flags.DEFINE_bool('viz', False, '')

# Geometry
tf.flags.DEFINE_integer('n_obj_pts', 1000, '')
tf.flags.DEFINE_integer('hand_size', 31, '')

# Model
tf.flags.DEFINE_integer('n_latent_factor', 64, '')
tf.flags.DEFINE_string('latent_factor_merge', 'concat', 'choose between "concat", "add"')
tf.flags.DEFINE_float('penetration_penalty', 0.03, '')
tf.flags.DEFINE_float('langevin_random_size', 0.1, '')
tf.flags.DEFINE_float('balancing_weight', 1e-4, '')
tf.flags.DEFINE_float('weight_decay', 1e-3, '')

# Train
tf.flags.DEFINE_integer('epochs', 10000, '')
tf.flags.DEFINE_integer('batch_size', 16, '')
tf.flags.DEFINE_float('lr_gen', 1e-3, '')
tf.flags.DEFINE_float('lr_des', 1e-3, '')
tf.flags.DEFINE_float('beta1_gen', 0.99, '')
tf.flags.DEFINE_float('beta1_des', 0.99, '')
tf.flags.DEFINE_float('beta2_gen', 0.999, '')
tf.flags.DEFINE_float('beta2_des', 0.999, '')
tf.flags.DEFINE_float('l2_reg', 0.001, '')

# Langevin
tf.flags.DEFINE_float('step_size', 0.1, '')
tf.flags.DEFINE_integer('langevin_steps', 20, '')
tf.flags.DEFINE_float('gradient_decay', 0.99, '')

flags = tf.flags.FLAGS