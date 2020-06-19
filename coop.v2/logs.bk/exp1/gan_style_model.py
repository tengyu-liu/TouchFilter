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
    self.lr_gen = config.lr_gen
    self.lr_des = config.lr_des
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
    self.obs_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'obs_hand')
    self.syn_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'syn_hand')
    self.obs_obj_rot = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'obs_obj_rot')
    self.obs_obj_trans = tf.placeholder(tf.float32, [self.batch_size, 3], 'obs_obj_trans')
    self.syn_z = tf.placeholder(tf.float32, [self.batch_size, self.n_latent_factor], 'syn_Z')
    self.obs_z = tf.placeholder(tf.float32, [self.batch_size, self.n_latent_factor], 'obs_Z')
    self.obj_id = tf.placeholder(tf.int32, [], 'obj_id')
    self.is_training = tf.placeholder(tf.bool, [], 'is_training')

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
      self.langevin_result = self.Langevin()
      self.obs_energy, self.obs_contact = self.Descriptor(self.obs_hand, self.obs_z, penetration_penalty=self.penetration_penalty)
      self.gen_energy, self.gen_contact = self.Descriptor(self.gen_hand, self.syn_z, penetration_penalty=self.penetration_penalty)
      self.qualified_candidates = self.gen_energy < self.obs_energy

  def build_train(self):
    print('[creating model] building train...')
    # self.gen_loss = tf.reduce_mean(tf.square(self.gen_hand - self.syn_hand))
    self.gen_loss = tf.reduce_mean(self.gen_energy)
    self.des_loss = tf.reduce_mean(tf.where(self.qualified_candidates, self.obs_energy - self.gen_energy, tf.zeros([self.batch_size])))
    self.reg_loss = tf.add_n(tf.get_collection('weight_decay'))
    # train Generator
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/gen')]
    # gen_optim = tf.train.AdamOptimizer(self.lr_gen, beta1=self.beta1_gen, beta2=self.beta2_gen)
    gen_optim = tf.train.GradientDescentOptimizer(self.lr_gen)
    gen_grads_vars = gen_optim.compute_gradients(self.gen_loss + self.reg_loss, var_list=gen_vars)
    self.train_gen = gen_optim.apply_gradients(gen_grads_vars)
    # train Descriptor
    des_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/des')]
    # des_optimizer = tf.train.AdamOptimizer(self.lr_des, beta1=self.beta1_des, beta2=self.beta2_des)
    des_optimizer = tf.train.GradientDescentOptimizer(self.lr_des)
    des_grads_vars = des_optimizer.compute_gradients(self.des_loss + self.reg_loss, var_list=des_vars)
    self.train_des = des_optimizer.apply_gradients(des_grads_vars)


  def build_summary(self):
    print('[creating model] building summary...')
    tf.summary.scalar('energy/obs', tf.reduce_mean(self.obs_energy))
    tf.summary.scalar('energy/gen', tf.reduce_mean(self.gen_energy))
    tf.summary.scalar('loss/gen', self.gen_loss)
    tf.summary.scalar('loss/des', self.des_loss)
    tf.summary.scalar('loss/reg', self.reg_loss)
    tf.summary.scalar('qualified_candidates', tf.reduce_mean(tf.cast(self.qualified_candidates, tf.float32)))
    tf.summary.histogram('contact/obs', self.obs_contact)
    tf.summary.histogram('contact/gen', self.gen_contact)
    self.summaries = tf.summary.merge_all()
    pass
  

  def Generator(self):
    with tf.variable_scope('gen'):
        h = pointnet_cls.get_model(self.obs_obj, is_training=self.is_training, n_latent_factor=self.n_latent_factor, weight_decay=self.weight_decay)
        if self.latent_factor_merge == 'concat':
          h = tf.concat([h, self.syn_z], axis=-1)
        elif self.latent_factor_merge == 'add':
          h = h + self.syn_z
        hand = bilinear(h, self.hand_size, scope='bilinear', num_hidden=256, is_training=self.is_training, weight_decay=self.weight_decay)
        hand = tf.concat([tf.clip_by_value(hand[:,:22], self.z_min, self.z_max), hand[:,22:]], axis=-1)
        return hand

  
  def Descriptor(self, hand, z, penetration_penalty=0):
    with tf.variable_scope('des'):
      # Compute hand surface point and normal
      jrot = hand[:,:22]
      grot = tf.reshape(hand[:,22:28], [hand.shape[0], 3, 2])
      gpos = hand[:,28:]
      hand_pts, hand_normals = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
      hand_pts = tf.concat(list(hand_pts.values()), axis=1)
      hand_normals = tf.concat(list(hand_normals.values()), axis=1)
      # transform hand_pts to obj coordinate
      hand_pts_for_obj = tf.transpose(tf.matmul(
        tf.transpose(self.obs_obj_rot, [0,2,1]), 
        tf.transpose(hand_pts - tf.expand_dims(self.obs_obj_trans, axis=1), [0,2,1])), [0,2,1])
      # Compute distance and angle between hand surface pts and obj surface
      hand_to_obj_dist, hand_to_obj_grad = tf.switch_case(self.obj_id-1, {
        0: lambda: self.obj_models[0].predict(hand_pts_for_obj),
        1: lambda: self.obj_models[1].predict(hand_pts_for_obj),
        2: lambda: self.obj_models[2].predict(hand_pts_for_obj),
        3: lambda: self.obj_models[3].predict(hand_pts_for_obj),
        4: lambda: self.obj_models[4].predict(hand_pts_for_obj),
        5: lambda: self.obj_models[5].predict(hand_pts_for_obj),
        6: lambda: self.obj_models[6].predict(hand_pts_for_obj),
        7: lambda: self.obj_models[7].predict(hand_pts_for_obj),
        8: lambda: self.obj_models[8].predict(hand_pts_for_obj),
        9: lambda: self.obj_models[9].predict(hand_pts_for_obj)})
      hand_to_obj_dist = tf.reshape(hand_to_obj_dist, [hand_pts.shape[0], -1, 1])  # B x N x 1
      # rotate hand_to_obj_gradient to global coordinate
      hand_to_obj_grad = tf.reshape(hand_to_obj_grad, [hand_pts.shape[0], -1, 3])
      hand_to_obj_grad = tf.transpose(tf.matmul(self.obs_obj_rot, tf.transpose(hand_to_obj_grad, [0,2,1])), [0,2,1])  
      # normalize directional vectors
      hand_normals /= tf.norm(hand_normals, axis=-1, keepdims=True)
      hand_to_obj_grad /= tf.norm(hand_to_obj_grad, axis=-1, keepdims=True)
      angles = tf.reduce_sum(hand_normals * hand_to_obj_grad, axis=-1, keepdims=True)
      # Compute contact/non-contact assignment
      hand_feat = tf.concat([hand_pts, hand_to_obj_dist, angles, self.hand_model.pts_feature], axis=-1)
      assignment = tf.nn.leaky_relu(pointnet_seg.get_model(hand_feat, z_feat=z, is_training=self.is_training, weight_decay=self.weight_decay)[0])
      # Compute energy according to contact assignment
      contact_energy = tf.nn.relu(-hand_to_obj_dist) + tf.nn.relu(hand_to_obj_dist) * tf.reduce_sum(hand_to_obj_grad * hand_normals, axis=-1, keepdims=True)
      # non_contact_energy = tf.nn.relu(hand_to_obj_dist) * penetration_penalty + 0.1
      # contact_non_contact_energies = tf.concat([contact_energy, non_contact_energy], axis=-1)  # B x N x 2
      penetration_energy = tf.nn.relu(hand_to_obj_dist) * penetration_penalty
      energies = assignment * contact_energy + penetration_energy # B x N x 2 
      prior = tf.reduce_sum(hand[:,:22] * hand[:,:22], axis=-1)
      return tf.reduce_sum(energies, axis=[1,2]) + prior, assignment


  def Langevin(self):
    e = self.Descriptor(self.syn_hand, self.syn_z, penetration_penalty=self.penetration_penalty)[0]
    grad_z = tf.gradients(e, [self.syn_z])[0]
    z = self.syn_z - self.step_size_square * grad_z + self.step_size * tf.random_normal(self.syn_z.shape, mean=0, stddev=self.langevin_random_size)
    return z, e