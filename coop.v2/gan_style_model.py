import tensorflow as tf

from utils import pointnet_cls, pointnet_seg, HandModel, CupModel
from utils.tf_util import *

class Model:
  def __init__(self, config, stats):
    self.build_config(config, stats)
    self.build_input()
    self.build_model()
    self.build_train()
    self.build_summary()


  def build_config(self, config, stats):
    print('[creating model] building config...')
    self.dtype = tf.float32
    self.batch_size = config.batch_size
    self.n_obj_pts = config.n_obj_pts
    self.hand_size = config.hand_size
    self.n_latent_factor = config.n_latent_factor
    self.latent_factor_merge = config.latent_factor_merge
    self.penetration_penalty = config.penetration_penalty
    self.langevin_steps = config.langevin_steps
    self.gradient_decay = config.gradient_decay
    self.langevin_random_size = config.langevin_random_size
    self.balancing_weight = config.balancing_weight
    self.beta1_gen = config.beta1_gen
    self.beta1_des = config.beta1_des
    self.beta2_gen = config.beta2_gen
    self.beta2_des = config.beta2_des
    self.step_size = tf.constant(config.step_size, dtype=self.dtype)
    self.step_size_square = self.step_size * self.step_size
    self.z_min = tf.constant(stats[0][:,:22], dtype=self.dtype)
    self.z_max = tf.constant(stats[1][:,:22], dtype=self.dtype)
    self.weight_decay = config.weight_decay
    # if config.restore_epoch >= 0:
    #   log_dir = os.path.join('logs', coonfig.name)
    #   restore_ema_path = os.path.join(log_dir, '%04d.ema.np'%(config.restore_epoch))
    #   self.ema_value = tf.constant(np.load(os.path.join(restore_ema_path)), dtype=dtype)
    # else:
    #   self.ema_value = tf.ones([1, self.hand_size], dtype=self.dtype)

  def build_input(self):
    print('[creating model] building input...')
    self.obs_obj = tf.placeholder(tf.float32, [self.batch_size,self.n_obj_pts,3], 'obs_obj')
    self.obs_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'obs_contact')
    self.obs_obj_rot = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'obs_obj_rot')
    self.obs_obj_trans = tf.placeholder(tf.float32, [self.batch_size, 3], 'obs_obj_trans')
    self.syn_z = tf.placeholder(tf.float32, [self.batch_size, self.n_latent_factor], 'syn_Z')
    self.obj_id = tf.placeholder(tf.int32, [], 'obj_id')
    self.is_training = tf.placeholder(tf.bool, [], 'is_training')
    self.lr_gen = tf.placeholder(tf.float32, [], 'lr_gen')
    self.lr_des = tf.placeholder(tf.float32, [], 'lr_des')

  def build_model(self):
    print('[creating model] building model...')
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      # Setup
      self.hand_model = HandModel.HandModel(batch_size=self.batch_size)
      self.obj_models = {0: CupModel.CupModel(1),
                          1: CupModel.CupModel(2),
                          2: CupModel.CupModel(3),
                          3: CupModel.CupModel(4),
                          4: CupModel.CupModel(5),
                          5: CupModel.CupModel(6),
                          6: CupModel.CupModel(7),
                          7: CupModel.CupModel(8),
                          8: CupModel.CupModel(9),
                          9: CupModel.CupModel(10)}
      # self.ema = EMA.EMA(decay=self.gradient_decay, dtype=self.dtype, value=self.ema_value)
      self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
      # Computation
      self.gen_hand = self.Generator()
      self.obs_contact = self.ComputeContact()
      self.obs_energy = self.Descriptor(self.obs_contact, penetration_penalty=self.penetration_penalty)
      self.gen_energy = self.Descriptor(self.gen_hand, penetration_penalty=self.penetration_penalty)
      self.qualified_candidates = self.gen_energy < self.obs_energy

  def build_train(self):
    print('[creating model] building train...')
    self.gen_loss = tf.reduce_mean(self.gen_energy)
    self.des_loss = tf.reduce_mean(tf.where(self.qualified_candidates, self.obs_energy - self.gen_energy, tf.zeros([self.batch_size])))
    self.reg_loss = tf.add_n(tf.get_collection('weight_decay'))
    # train Generator
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/gen')]
    # gen_optim = tf.train.AdamOptimizer(self.lr_gen, beta1=self.beta1_gen, beta2=self.beta2_gen)
    gen_optim = tf.train.GradientDescentOptimizer(self.lr_gen)
    gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
    self.train_gen = gen_optim.apply_gradients(gen_grads_vars)
    # train Descriptor
    des_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/des')]
    # des_optimizer = tf.train.AdamOptimizer(self.lr_des, beta1=self.beta1_des, beta2=self.beta2_des)
    des_optimizer = tf.train.GradientDescentOptimizer(self.lr_des)
    des_grads_vars = des_optimizer.compute_gradients(self.des_loss, var_list=des_vars)
    self.train_des = des_optimizer.apply_gradients(des_grads_vars)


  def build_summary(self):
    print('[creating model] building summary...')
    self.summ_oe = tf.placeholder(tf.float32, [None], 'obs_energy')
    self.summ_oe2 = tf.placeholder(tf.float32, [None], 'obs_energy_2')
    self.summ_ge = tf.placeholder(tf.float32, [None], 'gen_energy')
    self.summ_ge2 = tf.placeholder(tf.float32, [None], 'gen_energy2')
    self.summ_ge3 = tf.placeholder(tf.float32, [None], 'gen_energy3')
    tf.summary.scalar('qualified_candidates', tf.reduce_mean(tf.cast(self.summ_ge < self.summ_oe, tf.float32)))
    tf.summary.scalar('energy/obs', tf.reduce_mean(self.summ_oe))
    tf.summary.scalar('energy/gen', tf.reduce_mean(self.summ_ge))
    tf.summary.scalar('loss/des', tf.reduce_mean(self.summ_oe - self.summ_ge))
    tf.summary.scalar('loss/gen', tf.reduce_mean(self.summ_ge))
    tf.summary.scalar('impr/des_loss', tf.reduce_mean(self.summ_oe - self.summ_ge) - tf.reduce_mean(self.summ_oe2 - self.summ_ge2))
    tf.summary.scalar('impr/obs_energy', tf.reduce_mean(self.summ_oe) - tf.reduce_mean(self.summ_oe2))
    tf.summary.scalar('impr/gen_energy', tf.reduce_mean(self.summ_ge) - tf.reduce_mean(self.summ_ge3))
    tf.summary.scalar('impr/gen_energy_stage1', tf.reduce_mean(self.summ_ge) - tf.reduce_mean(self.summ_ge2))
    tf.summary.scalar('impr/gen_energy_stage2', tf.reduce_mean(self.summ_ge2) - tf.reduce_mean(self.summ_ge3))
    self.summaries = tf.summary.merge_all()
    pass
  
  # TODO: switch to a two-stage effort, where the prediction becomes contact point labling and then finger matching
  def Generator(self):
    with tf.variable_scope('gen'):
        # predict the contact point assignment
        contact_point_assignment = pointnet_seg.get_model(self.obs_obj, z_feat=self.syn_z, z_merge=self.latent_factor_merge, is_training=self.is_training, weight_decay=self.weight_decay)
        return contact_point_assignment

  def Descriptor(self, contact_point_assignment, penetration_penalty=0):
    with tf.variable_scope('des'):
      contact_point_data = tf.concat([self.obs_obj, contact_point_assignment], axis=-1)
      latent = pointnet_cls.get_model(contact_point_data, is_training=self.is_training, n_latent_factor=1024, weight_decay=self.weight_decay)
      energy = bilinear(latent, 1, 'latent', is_training=self.is_training, weight_decay=self.weight_decay)
      return energy[:,0]

  def ComputeContact(self):
    # TODO