import numpy as np
import tensorflow as tf

from config import config
from CupModel import CupModel

model = CupModel(config.cup_id, 199, 'models')

c = tf.ConfigProto()
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)

sess.run(tf.global_variables_initializer())

x = np.random.random([1000000, 3]) - 0.5
y, grad = sess.run([model.pred, model.grad], feed_dict={model.x: x})
c = np.logical_and(y[:,0] >= -0.02, y[:,0] <= -0.01)

import mayavi.mlab as mlab
# mlab.points3d(x[c,0], x[c,1], x[c,2], scale_factor=0.001)
# mlab.show()

import trimesh as tm
cup = tm.load_mesh('../onepiece/%d.obj'%config.cup_id)
mlab.triangular_mesh(cup.vertices[:,0], cup.vertices[:,1], cup.vertices[:,2], cup.faces)
mlab.points3d(x[c,0], x[c,1], x[c,2], scale_factor=0.001)
mlab.quiver3d(x[c,0], x[c,1], x[c,2], grad[c,0], grad[c,1], grad[c,2])
mlab.show()
