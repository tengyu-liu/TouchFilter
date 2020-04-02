import random
import sys
import datetime
import copy
import os
import pickle
import time
from collections import defaultdict

sys.path.append('..')

import imageio
import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import flags
from model import Model
from viz_util import Visualizer

flags.name = 'dynamic_z2_nobn_unitz2'
flags.restore_name = 'dynamic_z2_nobn_unitz2'
flags.restore_epoch = 99
flags.restore_batch = 300
flags.dynamic_z2 = True
flags.batch_size = 2
flags.adaptive_langevin = True
flags.clip_norm_langevin = True
flags.prior_type = 'NN'
flags.langevin_steps = 90
flags.step_size = 0.1

for k, v in flags.flag_values_dict().items():
    print(k, v)

try:
    project_root = os.path.join(os.path.dirname(__file__), '../..')
except:
    project_root = os.path.join('../..')

# load obj
cup_id_list = [3]
if flags.debug:
    cup_id_list = [1]

cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

obs_zs = defaultdict(list)
palm_directions = defaultdict(list)

all_zs = []

for i in cup_id_list:
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
        annotation = cup_annotation['%d_%d'%(i,j)]
        for frame in range(len(mat_data)):
            cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
            hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
            hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
            hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
            hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
            all_zs.append(hand_z)
        for start_end in annotation:
            start, end = [int(x) for x in start_end.split(':')]
            for frame in range(start, end):
                cup_id = i
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
                hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
                hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                palm_v = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotate([0,0,1])
                palm_directions[cup_id].append(palm_v)
                obs_zs[cup_id].append(hand_z)
                if flags.debug:
                    break
            if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
                break
        if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
            break

obs_zs = {i : np.array(x) for (i,x) in obs_zs.items()}
palm_directions = {i : np.array(x) for (i,x) in palm_directions.items()}
obs_z2s = {i : np.random.normal(size=[len(obs_zs[i]), flags.z2_size])}
all_zs = np.array(all_zs)

stddev = np.std(all_zs, axis=0, keepdims=True)
mean = np.mean(all_zs, axis=0, keepdims=True)
z_min = np.min(all_zs, axis=0, keepdims=True)
z_max = np.max(all_zs, axis=0, keepdims=True)

# load model
model = Model(flags, mean, stddev, cup_id_list)
print('Model loaded')

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# restore
saver = tf.train.Saver(max_to_keep=0)
if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join('../models', flags.restore_name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

# load training result
data = pickle.load(open('../figs/%s/%04d-%d.pkl'%(flags.name, flags.restore_epoch, flags.restore_batch), 'rb'))
print(data.keys())
_GT_syn_e = data['syn_e']
_GT_syn_z = data['syn_z']
_GT_syn_z2 = data['syn_z2']
_GT_syn_w = data['syn_w']
_GT_syn_p = data['syn_p']
_GT_g_avg = data['g_avg']

# prepare local data
update_mask = np.ones([flags.batch_size, 31])
update_mask[:,-9:-3] = 0.0    # We disallow grot update

"""
TODO: 
1. initialize z, z2 as gt initial data
2. fix z, interpolate z2 and draw contact activations
"""

syn_z = _GT_syn_z[:,0,:]
syn_z2 = _GT_syn_z2[:,0,:]

w = np.zeros([syn_z.shape[0], 10, 10, _GT_syn_w.shape[-1], 2])

v = Visualizer()
os.makedirs('figs', exist_ok=True)
for i_batch in range(syn_z.shape[0]):
    os.makedirs('figs/%d'%i_batch, exist_ok=True)

component = np.load('pca_components.npy')

def process(_GT_syn_z, _GT_syn_z2, sess, model, w_mean, w_std, i_batch, i_step, i_z2):
    cup_id = 3
    syn_z = _GT_syn_z[[i_batch,i_batch],i_step,:]
    _z2 = _GT_syn_z2[[i_batch,i_batch],i_step,:]
    _z2[0,:] -= component[0]
    _z2[1,:] += component[0]
    z, z2, syn_e, syn_w, syn_p = sess.run(model.syn_zzewpg[cup_id], feed_dict={
            model.inp_z: syn_z, model.inp_z2: _z2, model.update_mask: update_mask[[0,0]], model.is_training: False, model.gz_mean: _GT_g_avg})
    syn_w = syn_w[...,0]
    syn_w -= w_mean
    syn_w /= w_std
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    w1 = syn_w[0].copy()
    w2 = syn_w[1].copy()
    diff = w2 - w1
    v.visualize_weight(w1, ax=ax1, vmin=min(w1.min(), w2.min()), vmax=max(w1.max(), w2.max()))
    v.visualize_weight(w2, ax=ax2, vmin=min(w1.min(), w2.min()), vmax=max(w1.max(), w2.max()))
    v.visualize_weight(diff, ax=ax3)
    _ = ax4.hist(w1, bins=100, histtype='step')
    _ = ax4.hist(w2, bins=100, histtype='step')
    _ = ax4.hist(diff, bins=100, histtype='step')
    plt.show()
    return w1, w2, diff

w_mean = np.load('w_mean.npy')
w_std = np.load('w_std.npy')

w1, w2, diff = process(_GT_syn_z, _GT_syn_z2, sess, model, w_mean, w_std, 7, 0, [0])

# TODO: update z2 by pca components
