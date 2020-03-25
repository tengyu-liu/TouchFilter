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

from config import flags
from model import Model
from viz_util import Visualizer

flags.name = 'dynamic_z2_nobn_unitz2'
flags.restore_name = 'dynamic_z2_nobn_unitz2'
flags.restore_epoch = 99
flags.restore_batch = 300
flags.dynamic_z2 = True
flags.batch_size = 16
flags.adaptive_langevin = True
flags.clip_norm_langevin = True
flags.prior_type = 'NN'
flags.langevin_steps = 10000
flags.step_size = 0.001

for k, v in flags.flag_values_dict().items():
    print(k, v)

project_root = os.path.join(os.path.dirname(__file__), '../..')

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
    saver.restore(sess, os.path.join(os.path.dirname(__file__), '../models', flags.restore_name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

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

v = Visualizer()
os.makedirs('figs', exist_ok=True)
for i_batch in range(syn_z.shape[0]):
    os.makedirs('figs/%d'%i_batch, exist_ok=True)

for i_z2 in range(10):
    _z2 = syn_z2.copy()
    _z2[:,i_z2] -= 0.5
    for i_value in range(10):
        # run local synthesis
        _z2[:,i_z2] += 0.1
        z, z2, syn_e, syn_w, syn_p = sess.run(model.syn_zzewpg[cup_id], feed_dict={
            model.inp_z: syn_z, model.inp_z2: syn_z2, model.update_mask: update_mask, model.is_training: False, model.gz_mean: _GT_g_avg})
        for i_batch in range(syn_z.shape[0]):
            v.visualize_weight(3, syn_z[i_batch], syn_w[i_batch], 'figs/%d/%d-%d'%(i_batch, i_z2, i_value))