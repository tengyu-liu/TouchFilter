import numpy as np
import tensorflow as tf

class TouchFilter:
    def __init__(self, n_pts, n_filters=20, prange=0.5, pnum=50, pexp=2, nrange=-0.5, nnum=50, nexp=2, n_channel=2):
        self.filter_pos = tf.constant(np.concatenate([np.linspace(nrange,0,nnum) ** nexp * -1, np.linspace(0,prange, pnum) ** pexp]), dtype=tf.float32) # [100]
        self.n_pts = n_pts
        self.n_filters = n_filters
        self.n_channel = n_channel
        self.weights = tf.get_variable('des/touch/w', [1, 1, n_filters, pnum+nnum, self.n_channel], initializer=tf.random_normal_initializer(stddev=0.001), trainable=True)

    def __call__(self, pts, vectors, cup_model, cup_r):
        # pts: B x N x 3
        # vec: B x N x 3
        # input:  B x N x L
        # weight: 1, N x F x L
        # output: B x N x F x L
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

        if self.n_channel == 1:
            features = tf.expand_dims(dists, axis=-1)
        elif self.n_channel == 2:
            grads = cup_dists[...,1:]
            grads = tf.reshape(tf.transpose(tf.matmul(cup_r, tf.transpose(tf.reshape(grads, [filter_pts.shape[0], self.n_pts * self.filter_pos.shape[0], 3]), perm=[0,2,1])), perm=[0,2,1]), [filter_pts.shape[0], self.n_pts, 1, self.filter_pos.shape[0], 3])

            grads = grads / tf.norm(grads, axis=-1, keepdims=True)
            cosin = tf.reduce_sum(grads * tf.expand_dims(tf.expand_dims(vectors, axis=2), axis=2), axis=-1)
            features = tf.stack([dists, cosin], axis=-1)
        else:
            raise NotImplemented

        return tf.reduce_sum(tf.reduce_sum(self.weights * features, axis=-1), axis=-1)     # B x N x F

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
        