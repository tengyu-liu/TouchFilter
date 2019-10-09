import numpy as np
import tensorflow as tf

class TouchFilter:
    def __init__(self, n_pts, situation_invariant=False, penalty_strength=1e-2):
        self.n_pts = n_pts
        self.situation_invariant = situation_invariant
        self.penalty_strength = penalty_strength
        if self.situation_invariant:
            self.weight = tf.get_variable('des/touch/w', [1, self.n_pts, 2], initializer=tf.random_normal_initializer(stddev=0.001), trainable=True)
        else:
            self.dense_1 = tf.layers.Dense(1024, activation=tf.nn.relu)
            self.dense_2 = tf.layers.Dense(self.n_pts * 2)
        pass
        
    def __call__(self, pts, cup_model, cup_r, hand_z=None):
        # pts: B x N x 3
        pts = tf.concat(list(pts.values()), axis=1)
        filter_pts = tf.transpose(tf.matmul(
                tf.transpose(cup_r, perm=[0,2,1]), 
                tf.transpose(filter_pts, perm=[0,2,1])), perm=[0, 2, 1]) * 4
        dists = tf.reshape(cup_model.predict(tf.reshape(filter_pts, [-1,3])), [filter_pts.shape[0], -1, 4])  # B x N x 1

        f0 = tf.math.square(dists)
        f1 = tf.math.square(tf.nn.relu(dists))
        features = tf.stack([f0, f1], axis=-1)  # B x N x 2

        if self.situation_invariant:
            weight = self.weight
        else:
            situation = tf.concat([f0, hand_z], axis=-1)
            weight = tf.reshape(self.dense_2(self.dense_1(situation)), [1, -1, 2])
        
        weight = tf.nn.softmax(weight)
        energies = weight * features # B x N x 2 
        return tf.reduce_mean(tf.reduce_sum(energies, axis=[1,2]) + tf.reduce_sum(weight[...,1], axis=-1) * self.penalty_strength)

    def debug(self, pts, vectors, cup_model, cup_r):
        pts = tf.concat(list(pts.values()), axis=1)
        vectors = tf.concat(list(vectors.values()), axis=1)
        filter_pts = tf.expand_dims(pts, axis=2) + tf.expand_dims(vectors, axis=2) * tf.reshape(self.filter_pos, [1, 1, -1, 1]) # B x N x L x 3
        filter_pts_shape = filter_pts.shape
        filter_pts = tf.reshape(filter_pts, [filter_pts.shape[0], -1, 3])
        filter_pts = tf.transpose(tf.matmul(
                tf.transpose(cup_r, perm=[0,2,1]), 
                tf.transpose(filter_pts, perm=[0,2,1])), perm=[0, 2, 1]) * 4
        cup_dists = tf.reshape(cup_model.predict(tf.reshape(filter_pts, [-1,3])), [filter_pts.shape[0], self.n_pts, 1, self.filter_pos.shape[0], 4])  # B x N x 1 x L x 4
        dists = cup_dists[...,0]
        grads = cup_dists[...,1:]
        grads = tf.reshape(tf.transpose(tf.matmul(cup_r, tf.transpose(tf.reshape(grads, [filter_pts.shape[0], self.n_pts * self.filter_pos.shape[0], 3]), perm=[0,2,1])), perm=[0,2,1]), [filter_pts.shape[0], self.n_pts, 1, self.filter_pos.shape[0], 3])
        grads = grads / tf.norm(grads, axis=-1, keepdims=True)
        return dists, grads
        