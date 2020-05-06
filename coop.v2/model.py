import tensorflow as tf

class Model:

  """
  Pipeline:
    1. Generate h' using G
    2. Optimize C' and C^ to minimize E(h'|C') + E(C') and E(h^|C^) + E(C^)
    3. Optimize h* to minimize E(h'|C')
    4. Update E(C') to minimize E(C^) - E(C')
    5. Update G to produce h*
  """

  def __init__(self, config, stats):
    self.build_config(config, stats)
    self.build_input()
    self.build_model()
    self.build_train()
    self.build_summary()

  def build_config(self, config, stats):
    self.batch_size = config.batch_size
    self.z_size = config.z_size
    self.num_hand_points = config.num_hand_points
    self.hand_size = 31
    pass

  def build_input(self):
    self.obj_id = tf.placeholder(tf.int32, [])
    self.generator_z = tf.placeholder(self.dtype, [self.batch_size, self.z_size])
    self.langevin_contact_in = tf.placeholder(self.dtype, [self.batch_size, self.num_hand_points])
    self.langevin_hand_in = tf.placeholder(self.dtype, [self.batch_size, self.hand_size])
    self.obs_hand = tf.placeholder(self.dtype, [self.batch_size, self.hand_size])
    self.obs_contact_in = tf.placeholder(self.dtype, [self.batch_size, self.num_hand_points])
    pass
  
  def build_model(self):
    self.generated_hand = self.G()
    self.langevin_contact_out = self.LangevinContact()
    self.langevin_hand_out = self.LangevinHand()
    pass

  def ContactPrior(self, contact):
    pass

  def LangevinContact(self):
    e = self.D(self.langevin_hand_in, self.langevin_contact_in) + self.ContactPrior(self.langevin_contact_in)
    g = tf.gradients(e, self.langevin_contact_in)
    contact_out = self.langevin_contact_in - self.step_size_square * g + self.step_size * tf.random.normal(0, 1, self.langevin_contact_in.shape)
    return contact_out
  
  def LangevinHand(self):
    e = self.D(self.langevin_hand_in, self.langevin_contact_in)
    g = tf.gradients(e, self.langevin_hand_in)
    hand_out = self.langevin_hand_in - self.step_size_square * g + self.step_size * tf.random.normal(0, 1, self.langevin_hand_in.shape)
    return hand

  def D(self, hand, contact):
    pass

  def G(self):
    pass

  def build_train(self):

    pass

  def build_summary(self):
    pass