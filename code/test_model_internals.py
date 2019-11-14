import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion.quaternion import Quaternion as Q

from utils.CupModel import CupModel
from utils.vis_util import VisUtil
from utils.HandModel import HandModel

project_root = os.path.join(os.path.dirname(__file__), '..')
vu = VisUtil()

cup_id_list = [1,2,3,5,6,7,8]

cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

for i in cup_id_list:
    center = np.mean(cup_models[i].bounding_box.vertices, axis=0)
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
        annotation = cup_annotation['%d_%d'%(i,j)]
        for start_end in annotation:
            start, end = [int(x) for x in start_end.split(':')]
            for frame in range(start, end):
                cup_id = i
                cup_rotation = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotation_matrix
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
                hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                cup_rs[cup_id].append(cup_rotation)
                obs_zs[cup_id].append(hand_z)

cup_rs = {i:np.array(x) for (i,x) in cup_rs.items()}
obs_zs = {i:np.array(x) for (i,x) in obs_zs.items()}

tf_cup_models = { i: CupModel(i, 199, os.path.join(os.path.dirname(__file__), '../data/cups/models')) for i in cup_id_list }
tf_hand_model = HandModel(1)
tf_cup_r = tf.placeholder(tf.float32, [1,3,3], 'tf_cup_r')
tf_jrot = tf.placeholder(tf.float32, [1, 22], 'tf_jrot')
tf_grot = tf.placeholder(tf.float32, [1,3,2], 'tf_grot')
tf_gpos = tf.placeholder(tf.float32, [1,3], 'tf_gpos')
tf_pts, tf_normal = tf_hand_model.tf_forward_kinematics(tf_gpos, tf_grot, tf_jrot)
tf_pts = tf.concat(list(tf_pts.values()), axis=1)
tf_rot_pts = tf.transpose(tf.matmul(
                    tf.transpose(tf_cup_r, perm=[0,2,1]), 
                    tf.transpose(tf_pts, perm=[0,2,1])), perm=[0, 2, 1])
tf_dists_grads = {i : tf_cup_models[i].predict(tf.reshape(tf_rot_pts, [-1,3])) for i in cup_id_list}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

while True: 
    cup_id = random.sample(cup_id_list, 1)[0]
    item_id = np.random.randint(0, len(cup_rs[cup_id]) - 1)

    cup_r = cup_rs[cup_id][[item_id]]
    obs_z = obs_zs[cup_id][[item_id]]

    jrot = obs_z[:,:22]
    grot = obs_z[:,22:28].reshape([1,3,2])
    gpos = obs_z[:,28:]

    pts, dists = sess.run([tf_pts, tf_dists_grads[cup_id][0]], feed_dict={
        tf_cup_r: cup_r, tf_jrot: jrot, tf_grot: grot, tf_gpos: gpos
    })

    vu.visualize(cup_id, cup_r, obs_z)

    ax = plt.subplot(111, projection='3d')
    p = ax.scatter(pts[0,:,0], pts[0,:,1], pts[0,:,2], s=1, c=dists[:,0])
    plt.colorbar(p)
    plt.show()




"""
TODO: 
0. Run BodyNet
1. examine if dists and pointclouds and cup model are correct   [check]
2. try to fix bn in pointcloud (optional)
3. start from center, remove global rotation from update
"""