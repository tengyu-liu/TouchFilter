import copy
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab

from config import flags
from model import Model

from CupModel import CupModel
from HandModel import HandModel

from visualize.visualize_model_result import visualize

# load pca
pca_mean = np.load('data/pca_mean.npy')
pca_components = np.load('data/pca_components.npy')
pca_var = np.load('data/pca_variance.npy')
print('PCA loaded.')

# load obj
cup_id_list = [1,2,3,4,5,6,7,8]
cup_models = {cup_id: tm.load_mesh('data/cups/onepiece/%d.obj'%cup_id) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(os.path.dirname(__file__), 'data/cup_video_annotation.txt')).readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

for i in cup_id_list:
    center = np.mean(cup_models[i].bounding_box.vertices, axis=0)
    for j in range(1,11):
        mat_data = sio.loadmat('data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j))['glove_data']
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
                hand_jrot = np.stack([np.sin(hand_jrot), np.cos(hand_jrot)], axis=-1)
                hand_z = np.concatenate([hand_jrot.reshape([44]), hand_grot.reshape([6]), hand_gpos])
                hand_z = np.matmul((hand_z-pca_mean), pca_components.T) / np.sqrt(pca_var)

                cup_rs[cup_id].append(cup_rotation)
                obs_zs[cup_id].append(hand_z)

cup_rs = {i:np.array(x) for (i,x) in cup_rs.items()}
obs_zs = {i:np.array(x) for (i,x) in obs_zs.items()}

minimum_data_length = min(len(cup_rs[cup_id]) for cup_id in cup_id_list)
data_idxs = {cup_id: np.arange(len(cup_rs[cup_id])) for cup_id in cup_id_list}
batch_num = minimum_data_length // flags.batch_size * len(cup_id_list)
print('Training data loaded.')

# load model
pca_components = tf.constant(np.load('data/pca_components.npy'), dtype=tf.float32)
pca_mean = tf.constant(np.load('data/pca_mean.npy'), dtype=tf.float32)
pca_var = tf.constant(np.sqrt(np.expand_dims(np.load('data/pca_variance.npy'), axis=-1)), dtype=tf.float32)
hand_model = HandModel(1)
hand_x = tf.placeholder(tf.float32, [1,36])
z_ = tf.matmul(hand_x, pca_var * pca_components) + pca_mean
jrot = tf.reshape(z_[:,:44], [z_.shape[0], 22, 2])
grot = tf.reshape(z_[:,44:50], [z_.shape[0], 3, 2])
gpos = z_[:,50:]
obs_pts, _ = hand_model.tf_forward_kinematics(gpos, grot, jrot)

cup_models = { i : CupModel(i, 999, 'data/cups/models') for i in range(1,11)}
x = tf.placeholder(tf.float32, [None, 3])
cup_pred = {i : cup_models[i].predict(x) for i in range(1,11)}

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print('Start training...')

# train
shuffled_idxs = copy.deepcopy(data_idxs)
for cup_id in cup_id_list:
    np.random.shuffle(shuffled_idxs[cup_id])

for batch_id in range(batch_num):
    if batch_id < flags.restore_batch:
        continue
    
    t0 = time.time()
    cup_id = batch_id % len(cup_id_list) + 1
    item_id = batch_id // len(cup_id_list)
    idxs = shuffled_idxs[cup_id][flags.batch_size * item_id : flags.batch_size * (item_id + 1)]

    # load training data
    cup_r = cup_rs[cup_id][idxs]
    obs_z = obs_zs[cup_id][idxs]

    # compute hand_pts, cup_pts and cup_vals
    hand_pts = sess.run(obs_pts, feed_dict={hand_x: obs_z})
    hand_pts = np.concatenate(list(hand_pts.values()), axis=1)[0]

    center = np.mean(hand_pts, axis=0)

    cup_pts = np.random.random([100000,3]) * 0.6 - 0.3
    
    cup_pts_feed = np.matmul(cup_r[0].T, (cup_pts * 5).T).T
    
    cup_vals = sess.run(cup_pred[cup_id], feed_dict={x:cup_pts_feed})

    cup_vals[:,1:] = np.matmul(cup_r[0], cup_vals[:,1:].T).T

    keep = cup_vals[:,0] >= 0

    cup_pts = cup_pts[keep,:]
    cup_vals = cup_vals[keep]

    # draw with matplotlib
    mlab.clf()
    visualize(cup_id, cup_r[0], obs_z[0])
    mlab.points3d(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], mode='sphere', scale_factor=0.001, color=(0.,0.,0.))
    mlab.quiver3d(cup_pts[:,0], cup_pts[:,1], cup_pts[:,2], cup_vals[:,1], cup_vals[:,2], cup_vals[:,3])
    mlab.points3d(cup_pts[:,0], cup_pts[:,1], cup_pts[:,2], mode='sphere', scale_factor=0.001, color=(1,1,1))
    mlab.show()

    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], s=5, c='black')
    # ax.scatter(cup_pts[:,0], cup_pts[:,1], cup_pts[:,2], s=1, c=cup_vals)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    # # find cup vertices
    # cvert = copy.deepcopy(cup_models[cup_id].vertices)
    # cvert = np.matmul(cup_r[0], cvert.T).T
    
    # plot gt
    # mlab.clf()
    # mlab.points3d([-0.24,-0.24,-0.24,-0.24,0.24,0.24,0.24,0.24], [-0.24,-0.24,0.24,0.24,-0.24,-0.24,0.24,0.24], [-0.24,0.24,-0.24,0.24,-0.24,0.24,-0.24,0.24], mode='point', opacity=0.0)
    # mlab.triangular_mesh(cvert[:,0] / 0.8, cvert[:,1] / 0.8, cvert[:,2] / 0.8, cup_models[cup_id].faces, color=(0, 0, 1))

    # hand_pts = sess.run(model.obs_pts, feed_dict={model.obs_z: obs_z})
    # hand_pts = np.vstack([x[0] for x in hand_pts.values()])
    # mlab.points3d(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2])
    # mlab.savefig('figs/%s/%d-%d-gt.jpg'%(flags.name, epoch, batch_id))
    # mlab.show()

    # # plot initialization
    # mlab.clf()
    # mlab.points3d([-0.24,-0.24,-0.24,-0.24,0.24,0.24,0.24,0.24], [-0.24,-0.24,0.24,0.24,-0.24,-0.24,0.24,0.24], [-0.24,0.24,-0.24,0.24,-0.24,0.24,-0.24,0.24], mode='point', opacity=0.0)
    # mlab.triangular_mesh(cvert[:,0] / 0.8, cvert[:,1] / 0.8, cvert[:,2] / 0.8, cup_models[cup_id].faces, color=(0, 0, 1))

    # hand_pts = sess.run(model.obs_pts, feed_dict={model.obs_z: obs_z})
    # hand_pts = np.vstack([x[0] for x in hand_pts.values()])
    # mlab.points3d(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2])
    # mlab.savefig('figs/%s/%d-%d-init.jpg'%(flags.name, epoch, batch_id))

    # # plot synthesis
    # mlab.clf()
    # mlab.points3d([-0.24,-0.24,-0.24,-0.24,0.24,0.24,0.24,0.24], [-0.24,-0.24,0.24,0.24,-0.24,-0.24,0.24,0.24], [-0.24,0.24,-0.24,0.24,-0.24,0.24,-0.24,0.24], mode='point', opacity=0.0)
    # mlab.triangular_mesh(cvert[:,0] / 0.8, cvert[:,1] / 0.8, cvert[:,2] / 0.8, cup_models[cup_id].faces, color=(0, 0, 1))

    # hand_pts = sess.run(model.obs_pts, feed_dict={model.obs_z: obs_z})
    # hand_pts = np.vstack([x[0] for x in hand_pts.values()])
    # mlab.points3d(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2])
    # mlab.savefig('figs/%s/%d-%d-syn.jpg'%(flags.name, epoch, batch_id))

print()
