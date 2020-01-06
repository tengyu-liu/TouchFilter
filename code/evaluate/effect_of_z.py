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

from config import flags
from model import Model

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

data_idxs = {cup_id: np.arange(len(obs_zs[cup_id])) for cup_id in cup_id_list}
batch_num = 10  # Run experiments on 10 batches
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

for batch_id in range(batch_num):
    t0 = time.time()
    cup_id = cup_id_list[batch_id % len(cup_id_list)]
    item_id = batch_id // len(cup_id_list)
    idxs = shuffled_idxs[cup_id][item_id : item_id + 1] * flags.batch_size
    print(shuffled_idxs[cup_id][item_id : item_id + 1], flags.batch_size, idxs)
    obs_z = obs_zs[cup_id][idxs]
    syn_z = np.zeros(obs_z.shape)
    syn_z[:,:22] = 0
    syn_z[:,-9:] = obs_z[:,-9:]
    syn_z[:,-3:] += palm_directions[cup_id][idxs] * 0.1
    syn_z2_a = np.random.normal(size=[flags.z2_size])
    syn_z2_b = np.random.normal(size=[flags.z2_size])
    syn_z2 = np.linspace(syn_z2_a, syn_z2_b, flags.batch_size)

    syn_z_seq = np.zeros([flags.batch_size, 91, 31])
    syn_z2_seq = np.zeros([flags.batch_size, 91, flags.z2_size])
    syn_e_seq = np.zeros([flags.batch_size, 91, 1])
    syn_w_seq = np.zeros([flags.batch_size, 91, 5871])
    syn_p_seq = np.zeros([flags.batch_size, 91, 1])

    syn_z_seq[:,0,:] = syn_z
    syn_z2_seq[:,0,:] = syn_z2

    update_mask = np.ones(syn_z.shape)
    update_mask[:,-9:-3] = 0.0    # We disallow grot update

    for langevin_step in range(flags.langevin_steps):
        syn_z, syn_z2, syn_e, syn_w, syn_p, g_avg = sess.run(model.syn_zzewpg[cup_id], feed_dict={model.inp_z: syn_z, model.inp_z2: syn_z2, model.update_mask: update_mask, model.is_training: False})
        syn_z[:,:22] = np.clip(syn_z[:,:22], z_min[:,:22], z_max[:,:22])
        assert not np.any(np.isnan(syn_w))
        assert not np.any(np.isinf(syn_w))
        assert not np.any(np.isnan(syn_z))
        assert not np.any(np.isinf(syn_z))
        assert not np.any(np.isnan(syn_z2))
        assert not np.any(np.isinf(syn_z2))
        assert not np.any(np.isnan(syn_p))
        assert not np.any(np.isinf(syn_p))

        syn_z_seq[:, langevin_step+1, :] = syn_z
        syn_z2_seq[:, langevin_step+1, :] = syn_z2
        syn_e_seq[:, langevin_step, 0] = syn_e.reshape([-1])
        syn_w_seq[:, langevin_step, :] = syn_w[...,0]
        syn_p_seq[:, langevin_step, 0] = syn_p.reshape([-1])

    syn_ewp = sess.run(model.inp_ewp[cup_id], feed_dict={model.inp_z: syn_z, model.inp_z2: syn_z2, model.is_training: True})
    syn_e_seq[:, -1, 0] = syn_ewp[0].reshape([-1])
    syn_w_seq[:, -1, :] = syn_ewp[1][...,0]
    syn_p_seq[:, -1, 0] = syn_ewp[2].reshape([-1])

    pickle.dump({
        'syn_z_seq': syn_z_seq,
        'syn_z2_seq': syn_z2_seq,
        'syn_e_seq': syn_e_seq,
        'syn_p_seq': syn_p_seq,
        'syn_w_seq': syn_w_seq
    }, open('synthesis/%s-%04d-%d-%d.pkl'%(flags.name, flags.restore_epoch, flags.restore_batch, batch_id), 'wb'))

