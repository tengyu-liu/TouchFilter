import os
import tensorflow as tf

class CupModel:
    def __init__(self, cup_id, restore, weight_path):
        self.build_config(cup_id, restore, weight_path)
        self.build_input()
        self.build_model()

    def build_config(self, cup_id, restore, weight_path):
        self.float = tf.float32
        self.cup_id = cup_id
        self.num_layers = 8
        self.restore = restore
        self.weight_path = weight_path

    def build_input(self):
        with tf.variable_scope('cup%d'%self.cup_id):
            self.x = tf.placeholder(self.float, [None, 3])

    def predict(self, x):
        with tf.variable_scope('cup%d'%self.cup_id):
            h = x
            restore_filename = os.path.join(self.weight_path, 'exp_cup_%d__%d.ckpt'%(self.cup_id, self.restore))
            hand_reader = tf.train.NewCheckpointReader(restore_filename)
            for i in range(self.num_layers):
                if i == 0:
                    w = tf.Variable(hand_reader.get_tensor('dense/kernel'), trainable=False)
                    b = tf.Variable(hand_reader.get_tensor('dense/bias'), trainable=False)
                else:
                    w = tf.Variable(hand_reader.get_tensor('dense_%d/kernel'%i), trainable=False)
                    b = tf.Variable(hand_reader.get_tensor('dense_%d/bias'%i), trainable=False)
                h = tf.matmul(h, w) + b
                if i < self.num_layers - 1:
                    h = tf.nn.relu(h)
            
            # g = tf.gradients(h, x)[0]
            # g = g / tf.norm(g, axis=-1, keepdims=True)
            # out = tf.concat([h, g], axis=-1)
            return h

    def build_model(self):
        self.pred = self.predict(self.x)
        self.grad = tf.gradients(self.pred, self.x)[0]

if __name__ == "__main__":
    import numpy as np
    cm = CupModel(1, 29999, 'reduce_cup_model')
    axis = np.linspace(-0.3, 0.3, 100) * 5
    pts = np.transpose(np.stack(np.meshgrid(axis, axis, axis), axis=-1), [1,0,2,3]).reshape([-1,3])
    tf_pts = tf.constant(pts, dtype=tf.float32)
    tf_out = cm.predict(tf_pts)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dist = sess.run(tf_out)[:,0]
    print(pts.shape, dist.shape)
    # dist = grad[:,0]
    # grad = grad[:,1:]

    import mayavi.mlab as mlab

    mlab.points3d(pts[dist >= 0,0], pts[dist >= 0,1], pts[dist >= 0,2], scale_factor=0.05)
    # mlab.quiver3d(pts[::10,0], pts[::10,1], pts[::10,2], grad[::10,0], grad[::10,1], grad[::10,2])

    mlab.show()

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c=grad)
    # plt.show()