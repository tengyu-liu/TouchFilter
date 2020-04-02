import shutil
import sys
import datetime
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

sys.path.append('..')

from config import flags
from model import Model

flags.name = 'dynamic_z2_nobn_unitz2'
flags.restore_name = 'dynamic_z2_nobn_unitz2'
flags.restore_epoch = 99
flags.restore_batch = 300
flags.dynamic_z2 = True
flags.batch_size = 16
flags.adaptive_langevin = True
flags.clip_norm_langevin = True
flags.prior_type = 'NN'
flags.langevin_steps = 90
flags.step_size = 0.1

for k, v in flags.flag_values_dict().items():
    print(k, v)

f = open('history.txt', 'a')
f.write('[%s] python %s\n'%(str(datetime.datetime.now()), ' '.join(sys.argv)))
f.close()

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
gz_avg = pickle.load(open('../figs/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))['g_avg']

data_idxs = {cup_id: np.arange(len(obs_zs[cup_id])) for cup_id in cup_id_list}
batch_num = int(len(data_idxs[3]) // flags.batch_size)  # Run experiments on 10 batches
print('Training data loaded.')

# load model
model = Model(flags, mean, stddev, cup_id_list)
print('Model loaded')

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# load directories
log_dir = os.path.join(os.path.dirname(__file__), '../logs', flags.name)
model_dir = os.path.join(os.path.dirname(__file__), '../models', flags.name)
fig_dir = os.path.join(os.path.dirname(__file__), '../figs', flags.name)

# load checkpoint
saver = tf.train.Saver(max_to_keep=0)
if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(os.path.dirname(__file__), '../models', flags.name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

shuffled_idxs = copy.deepcopy(data_idxs)
for cup_id in cup_id_list:
    np.random.shuffle(shuffled_idxs[cup_id])

cup_id = 3
idxs = [0] * flags.batch_size
obs_z = obs_zs[cup_id][idxs]
syn_z = np.zeros([flags.batch_size, 31])
syn_z[:,:22] = 0
syn_z[:,-9:] = obs_z[:,-9:]
syn_z[:,-3:] += palm_directions[cup_id][idxs] * 0.1

syn_z2 = pickle.load(open('synthesis/same_z_diff_z2/%s/%04d-%d.pkl'%(flags.name, flags.restore_epoch, flags.restore_batch), 'rb'))['syn_z2']

update_mask = np.ones([flags.batch_size, 31])
update_mask[:,-9:-3] = 0.0    # We disallow grot update

# Step 1. Generate initial synthesis results
for langevin_step in range(flags.langevin_steps):
    syn_z, _, _, _, _ = sess.run(model.syn_zzewpg[cup_id], feed_dict={
            model.inp_z: syn_z, 
            model.inp_z2: syn_z2, 
            model.update_mask: update_mask, 
            model.gz_mean: gz_avg,
            model.is_training: False})

    syn_z[:,:22] = np.clip(syn_z[:,:22], z_min[:,:22], z_max[:,:22])
    # syn_z2 /= np.linalg.norm(syn_z2, axis=-1, keepdims=True)

    assert not np.any(np.isnan(syn_z))
    assert not np.any(np.isinf(syn_z))

syn_ewp = sess.run(model.inp_ewp[cup_id], feed_dict={
    model.inp_z: syn_z, 
    model.inp_z2: syn_z2, 
    model.gz_mean: gz_avg,
    model.is_training: True})
syn_e = syn_ewp[0].reshape([-1,1])
syn_w = syn_ewp[1][...,0]
syn_p = syn_ewp[2].reshape([-1,1])

os.makedirs('synthesis/same_z_diff_z2_level2', exist_ok=True)
os.makedirs('synthesis/same_z_diff_z2_level2/%s'%flags.name, exist_ok=True)

pickle.dump({
    'syn_z': syn_z,
    'syn_z2': syn_z2,
    'syn_e': syn_e,
    'syn_p': syn_p,
    'syn_w': syn_w
}, open('synthesis/same_z_diff_z2_level2/%s/%04d-%d.pkl'%(flags.name, flags.restore_epoch, flags.restore_batch), 'wb'))

