import os
import pickle

import numpy as np
import scipy.io as sio
import tensorflow as tf
from pyquaternion.quaternion import Quaternion as Q

from config import flags
from model import Model
from utils.tf_hand_kinematics import rotation_matrix


class MagnetizedModel:
  def __init__(self, batch_size):
    self.flags = flags
    self.flags.batch_size = batch_size
    mean = np.load('data/z_mean.npy')
    stddev = np.load('data/z_std.npy')
    self.z_min = np.load('data/z_min.npy')
    self.z_max = np.load('data/z_max.npy')
    self.gz_avg = pickle.load(open('data/model/0099-300.pkl', 'rb'))['g_avg']
    self.model = Model(flags, mean, stddev, [3])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=0)
    saver.restore(self.sess, 'data/model/0099-300.ckpt')

    cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(os.path.dirname(__file__), 'data/grasp/cup_video_annotation.txt')).readlines()}

    obs_zs = []
    palm_directions = []

    for i in [3]:
        for j in range(1,11):
            mat_data = sio.loadmat(os.path.join(os.path.dirname(__file__), 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
            annotation = cup_annotation['%d_%d'%(i,j)]
            for start_end in annotation:
                start, end = [int(x) for x in start_end.split(':')]
                for frame in range(start, end):
                    cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                    hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                    hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
                    hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
                    hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                    palm_v = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotate([0,0,1])
                    palm_directions.append(palm_v)
                    obs_zs.append(hand_z)
                    if flags.debug:
                        break
                if flags.debug and len(obs_zs) >= flags.batch_size:
                    break
            if flags.debug and len(obs_zs) >= flags.batch_size:
                break

    self.obs_zs = np.array(obs_zs)
    self.palm_directions = np.array(palm_directions)
    self.idx = np.arange(len(self.obs_zs))

    # parameter for magnetization
    self.alpha = tf.placeholder(tf.float32, [], 'alpha')
    self.noise_strength = tf.placeholder(tf.float32, [], 'noise_strength')
    # input for target
    self.target = tf.placeholder(tf.float32, [flags.batch_size, 31+flags.z2_size], 'target')
    # input for source
    self.X = tf.concat([self.model.inp_z, self.model.inp_z2], axis=-1)

    # distance between X and target
    self.distance = self.tf_distance(self.X, self.target)
    # energy + distance-to-target
    self.magnetized_energy = self.model.inp_ewp[3][0] + (self.distance[0] + self.distance[1] + self.distance[2] * 10) * self.alpha
    # gradient of magnetized_energy w.r.t. X
    self.magnetized_gradient = tf.gradients(tf.reduce_mean(self.magnetized_energy), [self.X])[0]
    
    # We don't allow grot update in sampling
    self.update_mask = np.ones([self.flags.batch_size, 31])
    self.update_mask[:,-9:-3] = 0.0    


  def sample_from_U(self):
    idxs = np.random.choice(self.idx, [self.flags.batch_size])
    obs_z = self.obs_zs[idxs]
    syn_z = np.zeros([self.flags.batch_size, 31])
    syn_z[:,:22] = 0
    syn_z[:,-9:] = obs_z[:,-9:]
    syn_z[:,-3:] += self.palm_directions[idxs] * 0.1
    syn_z2 = np.random.random([self.flags.batch_size, self.flags.z2_size])
    syn_z2 /= np.linalg.norm(syn_z2, axis=-1, keepdims=True)
    for langevin_step in range(self.flags.langevin_steps):
      syn_z, syn_z2, _, _, _, _ = self.sess.run(self.model.syn_zzewpg[3], feed_dict={
              self.model.inp_z: syn_z, 
              self.model.inp_z2: syn_z2, 
              self.model.update_mask: self.update_mask, 
              self.model.gz_mean: self.gz_avg,
              self.model.is_training: False})

      syn_z[:,:22] = np.clip(syn_z[:,:22], self.z_min[:,:22], self.z_max[:,:22])
      syn_z2 /= np.linalg.norm(syn_z2, axis=-1, keepdims=True)
    
    syn_ewp = self.sess.run(self.model.inp_ewp[3], feed_dict={
        self.model.inp_z: syn_z, 
        self.model.inp_z2: syn_z2, 
        self.model.gz_mean: self.gz_avg,
        self.model.is_training: False})

    return np.concatenate([syn_z, syn_z2], axis=-1), syn_ewp[0]

  def get_energy(self, x):
    return self.sess.run(self.model.inp_ewp[3], feed_dict={
      self.model.inp_z: x[:,:-self.flags.z2_size],
      self.model.inp_z2: x[:, -self.flags.z2_size:], 
      self.model.gz_mean: self.gz_avg, 
      self.model.is_training: False
    })[0]

  def tf_distance(self, x1, x2):
    jrot_dist = tf.reduce_sum(tf.abs(x1[:,:22] - x2[:,:22]), axis=-1)
    gpos_dist = tf.norm(x1[:,-3:] - x2[:,-3:], axis=-1)
    grot_x1 = rotation_matrix(tf.reshape(x1[:, :22:-3], [-1, 3, 2]))[:,:3,:3]
    grot_x2 = rotation_matrix(tf.reshape(x2[:, :22:-3], [-1, 3, 2]))[:,:3,:3]
    grot_R = tf.matmul(grot_x1, tf.transpose(grot_x2, [0,2,1]))
    grot_dist = tf.math.acos((tf.linalg.trace(grot_R) - 1) / 2)
    return jrot_dist, grot_dist, gpos_dist
  
  def get_distance(self, x1, x2):
    return self.sess.run(self.distance, feed_dict={
      self.model.inp_z: x1[:,:31], self.model.inp_z2: x1[:,31:], self.target: x2
    })

  def get_magnetized_gradient(self, x, target, alpha):
    return self.sess.run(self.magnetized_gradient, feed_dict={
      self.target: target, self.model.inp_z: x[:,:31], self.model.inp_z2: x[:,31:], self.alpha: alpha
    }) 
