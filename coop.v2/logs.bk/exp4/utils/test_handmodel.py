import numpy as np
import tensorflow as tf
from HandModel import HandModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hand_model = HandModel(1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

g_pos = np.zeros([1,3])
g_rot = np.random.random([1,6]).reshape([1,3,2])
j_rot = np.zeros([1,22])

pts, normal = sess.run(hand_model.tf_forward_kinematics(hand_model.gpos, hand_model.grot, hand_model.jrot), feed_dict={hand_model.gpos:g_pos, hand_model.grot: g_rot, hand_model.jrot: j_rot})
pts = np.concatenate(list(pts.values()), axis=1)[0]
normal = np.concatenate(list(normal.values()), axis=1)[0]

ax = plt.subplot(111, projection='3d')
ax.quiver(pts[:,0], pts[:,1], pts[:,2], normal[:,0], normal[:,1], normal[:,2], length=0.01)
plt.show()

# # normal /= np.linalg.norm(normal, axis=1, keepdims=True)
# normal += np.random.random(normal.shape) * 1e-4

# fig_data = [
#     # go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers'),
#     go.Cone(x=pts[:,0], y=pts[:,1], z=pts[:,2], u=normal[:,0], v=normal[:,1], w=normal[:,2])
# ]

# fig = go.Figure(data=fig_data)

# fig.show()

"""Conclusion: hand model normal points outwards"""