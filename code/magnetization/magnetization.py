"""
TODO: 
1. Get 1 z/z2 pair, run many steps of langevin, see if outcome is consistent
2. Get another z/z2 pair, run many steps of langevin, see if outcome is consistent and different
3. Run Magnetization from pair A to pair B
"""

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
from utils.vis_util import VisUtil

flags.name = 'dynamic_z2_nobn_unitz2'
flags.restore_name = 'dynamic_z2_nobn_unitz2'
flags.restore_epoch = 99
flags.restore_batch = 300
flags.dynamic_z2 = True
flags.batch_size = 1
flags.adaptive_langevin = True
flags.clip_norm_langevin = True
flags.prior_type = 'NN'
flags.langevin_steps = 250
flags.step_size = 0.02
flags.random_strength = 0.1
flags.prior_weight = 0.1

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

cup_id = 3

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

# define magetization loss
# mag. loss (from obs to inp) 
with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    tf_src_z = tf.placeholder(tf.float32, [flags.batch_size, 31], 'source_z')
    tf_src_z2 = tf.placeholder(tf.float32, [flags.batch_size, 10], 'source_z2')
    tf_tgt_z = tf.placeholder(tf.float32, [flags.batch_size, 31], 'target_z')
    tf_tgt_z2 = tf.placeholder(tf.float32, [flags.batch_size, 10], 'target_z2')
    tf_progress = tf.placeholder(tf.float32, [], 'progress')
    tf_dist_mult = tf.placeholder(tf.float32, [], 'dist_mult')
    tf_syn_z, tf_syn_z2, tf_curr_energy, tf_curr_weight, tf_curr_prior = model.syn_zzewpg[cup_id]
    tf_src_dist = tf.reduce_mean(tf.reduce_sum(tf.pow(tf_src_z2 - model.inp_z2, 2), axis=-1))
    tf_tgt_dist = tf.reduce_mean(tf.reduce_sum(tf.pow(tf_tgt_z2 - model.inp_z2, 2), axis=-1))
    tf_mag_loss = (tf_src_dist * (1 - tf_progress) + tf_tgt_dist * tf_progress) * tf_dist_mult + tf.reduce_mean(tf_curr_energy + tf_curr_prior * model.prior_weight)
    # tf_mag_loss = tf.reduce_mean(tf_curr_energy + tf_curr_prior * model.prior_weight)
    tf_z_grad, tf_z2_grad = tf.gradients(tf_mag_loss, [model.inp_z, model.inp_z2])

update_mask = np.ones([flags.batch_size, 31])
update_mask[:,-9:-3] = 0

def clip_by_norm(val, max_norm):
    norm = np.linalg.norm(val, axis=-1)
    clip = norm > max_norm
    clipped = val / np.expand_dims(norm, axis=-1) * max_norm
    val[clip,:] = clipped[clip,:]
    return val

import mayavi.mlab as mlab
vis = VisUtil()

for dist_mult in [10]:
    # load data
    data = pickle.load(open('synthesis/same_z_diff_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
    keep_ids = (data['syn_e'] < 3).reshape([-1])
    data['syn_w'] = data['syn_w'][keep_ids, :]
    data['syn_z'] = data['syn_z'][keep_ids, :]
    data['syn_z2'] = data['syn_z2'][keep_ids, :]
    data['syn_e'] = data['syn_e'][keep_ids, :]
    data['syn_p'] = data['syn_p'][keep_ids, :]

    for idx_A in [1]: #range(len(data['syn_z'])):
        for idx_B in [3]: #range(idx_A+1, len(data['syn_z'])):
            z_A, z_B = data['syn_z'][[idx_A, idx_B]].copy()
            z2_A, z2_B = data['syn_z2'][[idx_A, idx_B]].copy()

            num_steps = 1000
            step_size = 0.01
            random_scale = 0

            gz_avg = pickle.load(open('../figs/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))['g_avg']

            z_A = np.array([z_A] * flags.batch_size)
            z2_A = np.array([z2_A] * flags.batch_size)
            z_B = np.array([z_B] * flags.batch_size)
            z2_B = np.array([z2_B] * flags.batch_size)

            syn_z = z_A.copy()
            syn_z2 = z2_A.copy()

            syn_zs = np.zeros([flags.batch_size, num_steps + 1, 31])
            syn_z2s = np.zeros([flags.batch_size, num_steps + 1, 10])
            syn_es = np.zeros([flags.batch_size, num_steps + 1, 1])

            syn_zs[:,0,:] = syn_z
            syn_z2s[:,0,:] = syn_z2

            dist_to_tgt = np.zeros([flags.batch_size, num_steps + 1, 1])
            dist_to_src = np.zeros([flags.batch_size, num_steps + 1, 1])

            dist_to_src[:,0,0] = np.linalg.norm(syn_z2 - np.expand_dims(z2_A, axis=0), axis=-1)
            dist_to_tgt[:,0,0] = np.linalg.norm(syn_z2 - np.expand_dims(z2_B, axis=0), axis=-1)

            for step in range(num_steps):
                syn_z, syn_z2, src_dist, tgt_dist, energy, prior, loss = sess.run([
                    tf_syn_z, tf_syn_z2, tf_src_dist, tf_tgt_dist, tf_curr_energy, tf_curr_prior, tf_mag_loss
                ], feed_dict={
                    tf_src_z2: z2_A, tf_tgt_z2: z2_B, model.inp_z: syn_z, model.inp_z2: syn_z2, tf_progress: step / num_steps, 
                    model.is_training: False, tf_dist_mult: dist_mult, model.gz_mean: gz_avg, model.update_mask: update_mask
                })
                syn_z[:,:22] = np.clip(syn_z[:,:22], z_min[:,:22], z_max[:,:22])
                syn_z2 /= np.linalg.norm(syn_z2, axis=-1, keepdims=True)
                syn_zs[:,step + 1,:] = syn_z
                syn_z2s[:,step + 1,:] = syn_z2
                dist_to_src[:,step + 1,0] = np.linalg.norm(syn_z2 - np.expand_dims(z2_A, axis=0), axis=-1)
                dist_to_tgt[:,step + 1,0] = np.linalg.norm(syn_z2 - np.expand_dims(z2_B, axis=0), axis=-1)
                syn_es[:,step,0] = energy
                print('M', idx_A, idx_B, dist_mult, step, energy.mean(), prior.mean(), dist_to_src[:,step+1].mean(), dist_to_tgt[:,step+1].mean())
                # if step % 50 == 0:
                #     vis.visualize(3, syn_z[0])
                #     mlab.savefig('null.png')

            syn_es[:,-1,0] = sess.run(tf_curr_energy, feed_dict={
                model.inp_z: syn_z, model.inp_z2: syn_z2, tf_dist_mult: dist_mult, 
                model.is_training: False, model.gz_mean:gz_avg, model.update_mask: update_mask})

            pickle.dump({'z': syn_zs, 'z2': syn_z2s, 'e': syn_es, 'd_src': dist_to_src, 'd_tgt': dist_to_tgt}, open('mag_result_%d_%d_%f.pkl'%(idx_A, idx_B, dist_mult), 'wb'))
            exit()
