import numpy as np
import tensorflow as tf
from pointnet_seg import get_model as pointnet_model

class TouchFilter:
    def __init__(self, n_pts, situation_invariant=False, penalty_strength=1e-2):
        self.n_pts = n_pts
        self.situation_invariant = situation_invariant
        self.penalty_strength = penalty_strength
        if self.situation_invariant:
            self.weight = tf.get_variable('des/touch/w', [1, self.n_pts, 2], initializer=tf.random_normal_initializer(stddev=0.001), trainable=True)
        else:
            self.dense_1 = tf.layers.Dense(256, activation=tf.nn.relu)
            self.dense_2 = tf.layers.Dense(256, activation=tf.nn.relu)
        pass
        
    def __call__(self, pts, cup_model, cup_r, hand_z=None):
        # pts: B x N x 3
        pts = tf.transpose(tf.matmul(
                tf.transpose(cup_r, perm=[0,2,1]), 
                tf.transpose(pts, perm=[0,2,1])), perm=[0, 2, 1]) * 4
        dists = tf.reshape(cup_model.predict(tf.reshape(pts, [-1,3])), [pts.shape[0], -1, 1])  # B x N x 1

        f0 = tf.math.square(dists)
        f1 = tf.math.square(tf.nn.relu(dists))
        features = tf.concat([f0, f1], axis=-1)  # B x N x 2

        if self.situation_invariant:
            weight = self.weight
        else:
            z_feat = self.dense_2(self.dense_1(hand_z))
            weight = pointnet_model(dists, z_feat=z_feat)[0]
        
        weight = tf.nn.softmax(weight)
        energies = weight * features # B x N x 2 
        return tf.reduce_mean(tf.reduce_sum(energies, axis=[1,2]) + tf.reduce_sum(weight[...,1], axis=-1) * self.penalty_strength), weight
