import os
from config import flags
from utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import viz_util

import tensorflow as tf

from utils import pointnet_cls, pointnet_seg, HandModel, CupModel
from utils.tf_util import *

class Model:
  def __init__(self, config, stats):
    self.build_config(config, stats)
    self.build_input()
    self.build_model()


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
    # if config.restore_epoch >= 0:
    #   log_dir = os.path.join('logs', coonfig.name)
    #   restore_ema_path = os.path.join(log_dir, '%04d.ema.np'%(config.restore_epoch))
    #   self.ema_value = tf.constant(np.load(os.path.join(restore_ema_path)), dtype=dtype)
    # else:
    #   self.ema_value = tf.ones([1, self.hand_size], dtype=self.dtype)

  def build_input(self):
    print('[creating model] building input...')
    self.obs_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'obs_hand')
    self.obs_obj_rot = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'obs_obj_rot')
    self.obs_obj_trans = tf.placeholder(tf.float32, [self.batch_size, 3], 'obs_obj_trans')
    self.obj_id = tf.placeholder(tf.int32, [], 'obj_id')

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
      self.output = self.Descriptor(self.obs_hand)

  def Descriptor(self, hand, penetration_penalty=0):
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
      """
      FIXME: Need to be tested:
              1. transform hand pts to obj coordinate
              2. rotate hand_to_obj_gradient to global_coordinate
      """
      # normalize directional vectors
      hand_normals /= tf.norm(hand_normals, axis=-1, keepdims=True)
      hand_to_obj_grad /= tf.norm(hand_to_obj_grad, axis=-1, keepdims=True)
      return hand_pts, hand_normals, hand_to_obj_dist, hand_to_obj_grad, hand_pts_for_obj

      


vis = viz_util.Visualizer()

import plotly.graph_objects as go
from config import flags
import trimesh as tm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dl = DataLoader(flags)
model = Model(flags, [dl.z_min, dl.z_max])
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for obj_id, item_id, obs_z, obs_z2, obs_obj, obj_trans, obj_rot, idx in dl.fetch():
    print(obj_id, item_id)
    cup_model = tm.load(os.path.join(os.path.dirname(__file__), '../data/cups/onepiece/%d.obj'%obj_id))
    hand_pts, hand_normals, hand_to_obj_dist, hand_to_obj_grad, hand_pts_for_obj = sess.run(model.output, feed_dict={
      model.obj_id: obj_id, model.obs_hand: obs_z, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans})
    for i in range(len(obs_z)):
        vis.visualize_weight(obj_id, obs_z[i], obj_rot[i], obj_trans[i], 0)
        input()
        # break
        # # draw hand_to_obj_dist
        # input()
        # hand_to_obj_dist[i] -= np.min(hand_to_obj_dist[i])
        # hand_to_obj_dist[i] /= np.max(hand_to_obj_dist[i])
        # print(hand_to_obj_dist[i])
        # fig = go.Figure(data=fig_data)
        # fig.show()
        # input()
        # # draw hand normal
        # print(hand_normals[i])
        # fig_data = [go.Scatter3d(x=hand_pts[i,:,0], y=hand_pts[i,:,1], z=hand_pts[i,:,2], mode='markers'), 
        #             go.Cone(x=hand_pts[i,:,0], y=hand_pts[i,:,1], z=hand_pts[i,:,2], u=hand_normals[i,:,0], v=hand_normals[i,:,1], w=hand_normals[i,:,2])]
        # fig = go.Figure(data=fig_data)
        # fig.show()
        # input()
        # hand_to_obj_grad[i] *= 0.01
        # ax = plt.subplot(111, projection='3d')
        # ax.quiver(hand_pts[i,::10,0], hand_pts[i,::10,1], hand_pts[i,::10,2], hand_to_obj_grad[i,::10,0], hand_to_obj_grad[i,::10,1], hand_to_obj_grad[i,::10,2])
        # plt.show()
        # # draw obj normal
        # print(hand_to_obj_grad[i])
        # fig_data = [go.Cone(x=hand_pts[i,:,0], y=hand_pts[i,:,1], z=hand_pts[i,:,2], u=hand_to_obj_grad[i,:,0], v=hand_to_obj_grad[i,:,1], w=hand_to_obj_grad[i,:,2])]
        # fig = go.Figure(data=fig_data)
        # fig.show()
        # input()