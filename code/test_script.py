import numpy as np
import tensorflow as tf
from pointnet_seg import get_model

pts = tf.constant(np.random.random([100,1000,1]), dtype=tf.float32)
z_feat = tf.constant(np.random.random([100, 1024]), dtype=tf.float32)
is_training = tf.constant(True)

a = get_model(pts, is_training, z_feat)
b = get_model(pts, is_training, z_feat)
c = get_model(pts, is_training, z_feat)
d = get_model(pts, is_training, z_feat)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

f = open('log.txt', 'w')
for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    f.write(k.name + '\n')
    
f.close()

# We want 38 EMA variables