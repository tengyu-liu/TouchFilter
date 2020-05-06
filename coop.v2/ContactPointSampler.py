"""
With a descriptor D that computes E(H|C), use Swendsen-Wang method to sample the best contact-point assignment for observed examples. 
"""
import sys
import random

import numpy as np
import tensorflow as tf

sys.path.append('../coop_code/')
from utils import HandModel, CupModel
from utils.viz_util import Visualizer
from utils.data import DataLoader

class Sampler:
  def __init__(self):
    self.batch_size = 1
    self.hand_size = 31
    self.penetration_penalty = 0
    self.non_contact_penalty = 0.1

    self.obj_id = tf.placeholder(tf.int32, [], 'obj_id')
    self.obs_hand = tf.placeholder(tf.float32, [self.batch_size,self.hand_size], 'obs_hand')
    self.contact_points = tf.placeholder(tf.float32, [self.batch_size, 5871], 'contact_points')

    self.hand_model = HandModel.HandModel(batch_size=self.batch_size)
    self.obj_models = {i: CupModel.CupModel(i) for i in range(1, 11)}

    self.energy = self.descriptor(self.obs_hand, self.obj_id, self.contact_points)
    pass
  
  def descriptor(self, hand, obj_id, contact_points):
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
      hand_to_obj_dist = tf.reshape(hand_to_obj_dist, [hand_pts.shape[0], -1])
      hand_to_obj_grad = tf.reshape(hand_to_obj_grad, [hand_pts.shape[0], -1, 3])
      hand_normals /= tf.norm(hand_normals, axis=-1, keepdims=True)
      hand_to_obj_grad /= tf.norm(hand_to_obj_grad, axis=-1, keepdims=True)
      # compute energy
      contact_energy = tf.nn.relu(-hand_to_obj_dist) + tf.nn.relu(hand_to_obj_dist) * tf.reduce_sum(hand_to_obj_grad * hand_normals, axis=-1, keepdims=False)
      non_contact_energy = tf.nn.relu(hand_to_obj_dist) + self.non_contact_penalty
      energies = contact_energy * contact_points + non_contact_energy * (1 - contact_points)
      return tf.reduce_sum(energies, axis=[1])
  

if __name__ == '__main__':
  from config import flags
  flags.batch_size = 1
  dataloader = DataLoader(flags)
  num_points = 5871

  sampler = Sampler()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())


  for obj_id, item_id, hand, obj_pts in dataloader.fetch():
    energies = []
    contact_points = np.random.randint(0, 1, [1, num_points])
    energy = sess.run(sampler.energy, feed_dict={sampler.obs_hand: hand, sampler.obj_id: 3, sampler.contact_points: contact_points})
    for idx in range(num_points):
      print('\r [%d]: %f'%(idx, energy))
      new_cp = contact_points.copy()
      new_cp[0, idx] = 1 - new_cp[0, idx]
      new_energy = sess.run(sampler.energy, feed_dict={sampler.obs_hand: hand, sampler.obj_id: 3, sampler.contact_points: new_cp})
      if new_energy < energy:
        contact_points[0,idx] = 1 - contact_points[0,idx]
        energy = new_energy
      energies.append(energy)
  
    visualizer = Visualizer()
    visualizer.visualize_weight(hand[0], contact_points[0])
    import matplotlib.pyplot as plt
    plt.plot(energies)
    plt.show()