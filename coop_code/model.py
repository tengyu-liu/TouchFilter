import tensorflow as tf

from utils import pointnet_cls, pointnet_seg, HandModel, CupModel, EMA
from utils.tf_util import *

class Model:
  def __init__(self, config):
    self.build_config(config)
    self.build_input()
    self.build_model()
    self.build_train()
    self.build_summary()


  def build_config(self, config):
    print('[creating model] building config...')
    self.dtype = tf.float32
    self.batch_size = config.batch_size
    self.n_obj_pts = config.n_obj_pts
    self.hand_size = config.hand_size
    self.n_latent_factor = config.n_latent_factor
    self.penetration_penalty = config.penetration_penalty
    self.langevin_steps = config.langevin_steps
    self.gradient_decay = config.gradient_decay
    self.langevin_random_size = config.langevin_random_size
    self.lr_gen = config.lr_gen
    self.lr_des = config.lr_des
    self.beta1_gen = config.beta1_gen
    self.beta1_des = config.beta1_des
    self.beta2_gen = config.beta2_gen
    self.beta2_des = config.beta2_des
    self.step_size = tf.constant(config.step_size, dtype=self.dtype)
    self.step_size_square = self.step_size * self.step_size
    self.ema_value = None
    if config.restore_epoch >= 0:
      log_dir = os.path.join('logs', coonfig.name)
      restore_ema_path = os.path.join(log_dir, '%04d.ema.np'%(config.restore_epoch))
      self.ema_value = tf.constant(np.load(os.path.join(restore_ema_path)), dtype=dtype)


  def build_input(self):
    print('[creating model] building input...')
    self.obs_obj = tf.placeholder(tf.float32, [self.batch_size,self.n_obj_pts,3], 'obs_obj')
    self.obs_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'obs_hand')
    self.syn_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'syn_hand')
    self.Z = tf.placeholder(tf.float32, [self.batch_size, self.n_latent_factor], 'Z')
    self.obj_id = tf.placeholder(tf.int32, [], 'obj_id')
    self.is_training = tf.placeholder(tf.bool, [], 'is_training')

  def build_model(self):
    print('[creating model] building model...')
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      # Setup
      self.hand_model = HandModel.HandModel(batch_size=self.batch_size)
      self.obj_models = {i: CupModel.CupModel(i) for i in range(1, 11)}
      self.ema = EMA.EMA(decay=self.gradient_decay, size=[1, self.hand_size], dtype=self.dtype, value=self.ema_value)
      # Computation
      self.gen_hand = self.Generator()
      self.langevin_result = self.Langevin()
      self.obs_energy, self.obs_contact = self.Descriptor(self.obs_hand, penetration_penalty=0)
      self.syn_energy, self.syn_contact = self.Descriptor(self.syn_hand, penetration_penalty=self.penetration_penalty)
      self.gen_energy, self.gen_contact = self.Descriptor(self.gen_hand, penetration_penalty=self.penetration_penalty)


  def build_train(self):
    print('[creating model] building train...')
    self.gen_loss = tf.reduce_mean(tf.square(self.gen_hand - self.syn_hand))
    self.des_loss = tf.reduce_mean(self.obs_energy - self.syn_energy)
    # train Generator
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/gen')]
    gen_optim = tf.train.AdamOptimizer(self.lr_gen, beta1=self.beta1_gen, beta2=self.beta2_gen)
    gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
    self.train_gen = gen_optim.apply_gradients(gen_grads_vars)
    # train Descriptor
    des_vars = [var for var in tf.trainable_variables() if var.name.startswith('model/des')]
    des_optimizer = tf.train.AdamOptimizer(self.lr_des, beta1=self.beta1_des, beta2=self.beta2_des)
    des_grads_vars = des_optimizer.compute_gradients(self.des_loss, var_list=des_vars)
    self.train_des = des_optimizer.apply_gradients(des_grads_vars)


  def build_summary(self):
    print('[creating model] building summary...')
    tf.summary.scalar('energy/obs', tf.reduce_mean(self.obs_energy))
    tf.summary.scalar('energy/syn', tf.reduce_mean(self.syn_energy))
    tf.summary.scalar('energy/improve', tf.reduce_mean(self.syn_energy - self.gen_energy))
    tf.summary.scalar('loss/gen', self.gen_loss)
    tf.summary.scalar('loss/des', self.des_loss)
    tf.summary.histogram('contact/obs', self.obs_contact)
    tf.summary.histogram('contact/syn', self.syn_contact)
    tf.summary.histogram('contact/gen', self.gen_contact)
    self.summaries = tf.summary.merge_all()
    pass
  

  def Generator(self):
    with tf.variable_scope('gen'):
        h = pointnet_cls.get_model(self.obs_obj, is_training=self.is_training)
        h = tf.concat([h, self.Z], axis=-1)
        hand = bilinear(h, self.hand_size, scope='bilinear', num_hidden=512, is_training=self.is_training)
        return hand
  

  def Descriptor(self, hand, penetration_penalty=0):
    with tf.variable_scope('des'):
      # Compute hand surface point and normal
      jrot = hand[:,:22]
      grot = tf.reshape(hand[:,22:28], [hand.shape[0], 3, 2])
      gpos = hand[:,28:]
      hand_pts, hand_normals = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
      hand_pts = tf.concat(list(hand_pts.values()), axis=1)
      hand_normals = tf.concat(list(hand_normals.values()), axis=1)
      # Compute distance and angle between hand surface pts and obj surface
      hand_to_obj_dist, hand_to_obj_grad = tf.switch_case(self.obj_id, {i: lambda: self.obj_models[i+1].predict(hand_pts) for i in [j-1 for j in self.obj_models.keys()]})
      hand_to_obj_dist = tf.reshape(hand_to_obj_dist, [hand_pts.shape[0], -1, 1])  # B x N x 1
      hand_to_obj_grad = tf.reshape(hand_to_obj_grad, [hand_pts.shape[0], -1, 3])
      hand_normals /= tf.norm(hand_normals, axis=-1, keepdims=True)
      hand_to_obj_grad /= tf.norm(hand_to_obj_grad, axis=-1, keepdims=True)
      angles = tf.reduce_sum(hand_normals * hand_to_obj_grad, axis=-1, keepdims=True)
      # Compute contact/non-contact assignment
      hand_feat = tf.concat([hand_pts, hand_to_obj_dist, angles, self.hand_model.pts_feature], axis=-1)
      assignment = pointnet_seg.get_model(hand_feat, is_training=self.is_training)[0]
      # Compute energy according to contact assignment
      f0 = tf.nn.relu(-hand_to_obj_dist) + tf.nn.relu(hand_to_obj_dist) * penetration_penalty
      f1 = tf.nn.relu(hand_to_obj_dist) * penetration_penalty
      features = tf.concat([f0, f1], axis=-1)  # B x N x 2
      assignment = tf.reshape(tf.nn.softmax(tf.reshape(assignment, [-1, 2])), assignment.shape)
      energies = assignment * features # B x N x 2 
      return tf.reduce_sum(energies, axis=[1,2]), assignment[:,:,0]
      

  def Langevin(self):
    e = self.Descriptor(self.syn_hand)
    grad = tf.gradients(e, self.syn_hand)[0]
    self.ema.update(grad)
    grad /= self.ema.get()
    hand = self.syn_hand - self.step_size_square * grad + self.step_size * tf.random_normal(self.syn_hand.shape, mean=0, stddev=self.langevin_random_size)
    return hand, e