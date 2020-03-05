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
# from mayavi import mlab

from config import flags
from model import Model
from utils.EMA import EMA

if flags.tb_render:
    from utils.vis_util import VisUtil
    # load vis_util
    vu = VisUtil()

for k, v in flags.flag_values_dict().items():
    print(k, v)

f = open('history.txt', 'a')
f.write('[%s] python %s\n'%(str(datetime.datetime.now()), ' '.join(sys.argv)))
f.close()

project_root = os.path.join(os.path.dirname(__file__), '..')

# load obj
cup_id_list = [1,2,3,4,5,6]
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

gradient_ema = EMA(decay=0.99, size=[31])

minimum_data_length = min(len(obs_zs[cup_id]) for cup_id in cup_id_list)
data_idxs = {cup_id: np.arange(len(obs_zs[cup_id])) for cup_id in cup_id_list}
batch_num = minimum_data_length // flags.batch_size * len(cup_id_list)
print('Training data loaded.')

# load model
model = Model(flags, mean, stddev, cup_id_list)
print('Model loaded')

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# create directories
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs', flags.name), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models', flags.name), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'figs'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'figs', flags.name), exist_ok=True)
log_dir = os.path.join(os.path.dirname(__file__), 'logs', flags.name)
model_dir = os.path.join(os.path.dirname(__file__), 'models', flags.name)
fig_dir = os.path.join(os.path.dirname(__file__), 'figs', flags.name)

shutil.copy('train.py', os.path.join(log_dir, 'train.py'))
shutil.copy('model.py', os.path.join(log_dir, 'model.py'))
shutil.copy('config.py', os.path.join(log_dir, 'config.py'))

f = open(os.path.join(log_dir, 'command.txt'), 'w')
f.write('[%s] python %s\n'%(str(datetime.datetime.now()), ' '.join(sys.argv)))
f.close()

# logger and saver
train_writer = tf.summary.FileWriter(log_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=0)
if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'models', flags.restore_name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

print('Start training...')

# train
for epoch in range(flags.epochs):
    if epoch < flags.restore_epoch:
        continue

    shuffled_idxs = copy.deepcopy(data_idxs)
    for cup_id in cup_id_list:
        np.random.shuffle(shuffled_idxs[cup_id])
    
    for batch_id in range(batch_num):
        if epoch == flags.restore_epoch and batch_id <= flags.restore_batch:
            continue
        
        t0 = time.time()
        obs_z, syn_z, obs_z2, syn_z2, syn_z_seq, syn_z2_seq, obs_z2_seq, syn_e_seq, syn_w_seq, syn_p_seq = {},{},{},{},{},{},{},{},{},{}
        for cup_id in cup_id_list:
            idxs = shuffled_idxs[cup_id][flags.batch_size * item_id : flags.batch_size * (item_id + 1)]

            obs_z[cup_id] = obs_zs[cup_id][idxs]
            syn_z[cup_id] = np.zeros(obs_z[cup_id].shape)
            syn_z[cup_id][:,:22] = 0
            syn_z[cup_id][:,-9:] = obs_z[cup_id][:,-9:]
            syn_z[cup_id][:,-3:] += palm_directions[cup_id][idxs] * 0.1
            syn_z2[cup_id] = np.random.normal(size=[flags.batch_size, flags.z2_size])
            obs_z2[cup_id] = np.random.normal(size=[flags.batch_size, flags.z2_size])
            syn_z2[cup_id] /= np.linalg.norm(syn_z2[cup_id], axis=-1, keepdims=True)
            obs_z2[cup_id] /= np.linalg.norm(obs_z2[cup_id], axis=-1, keepdims=True)

            syn_z_seq[cup_id] = np.zeros([flags.batch_size, 91, 31])
            syn_z2_seq[cup_id] = np.zeros([flags.batch_size, 91, flags.z2_size])
            obs_z2_seq[cup_id] = np.zeros([flags.batch_size, 91, flags.z2_size])
            syn_e_seq[cup_id] = np.zeros([flags.batch_size, 91, 1])
            syn_w_seq[cup_id] = np.zeros([flags.batch_size, 91, 5871])
            syn_p_seq[cup_id] = np.zeros([flags.batch_size, 91, 1])

            syn_z_seq[cup_id][:,0,:] = syn_z[cup_id]
            syn_z2_seq[cup_id][:,0,:] = syn_z2[cup_id]
            obs_z2_seq[cup_id][:,0,:] = obs_z2[cup_id]

        update_mask = np.ones(syn_z[cup_id].shape)
        update_mask[:,-9:-3] = 0.0    # We disallow grot update

        for langevin_step in range(flags.langevin_steps):
            feed_dict = {model.update_mask: update_mask, model.is_training: False, model.g_avg: gradient_ema.get()}
            for cup_id in cup_id_list:
                feed_dict[model.inp_z[cup_id]] = syn_z[cup_id]
                feed_dict[model.inp_z2[cup_id]] = syn_z2[cup_id]            
            langevin_result_syn = sess.run(model.syn_zzewpg, feed_dict=feed_dict)
            feed_dict = {model.update_mask: update_mask, model.is_training: False, model.g_avg: gradient_ema.get()}
            for cup_id in cup_id_list:
                syn_z[cup_id] = langevin_result_syn[cup_id][0]
                syn_z2[cup_id] = langevin_result_syn[cup_id][1]
                syn_e[cup_id] = langevin_result_syn[cup_id][2]
                syn_w[cup_id] = langevin_result_syn[cup_id][3]
                syn_p[cup_id] = langevin_result_syn[cup_id][4]
                feed_dict[model.inp_z[cup_id]] = obs_z[cup_id]
                feed_dict[model.inp_z2[cup_id]] = obs_z2[cup_id]
                gradient_ema.apply(langevin_result_syn[cup_id][5])

            langevin_result_obs = sess.run(model.syn_zzewpg, feed_dict=feed_dict)
            for cup_id in cup_id_list:
                obs_z2[cup_id] = langevin_result_obs[cup_id][1]
                syn_z[cup_id][:,:22] = np.clip(syn_z[cup_id][:,:22], z_min[:,:22], z_max[:,:22])
                syn_z2[cup_id] /= np.linalg.norm(syn_z2[cup_id], axis=-1, keepdims=True)
                obs_z2[cup_id] /= np.linalg.norm(obs_z2[cup_id], axis=-1, keepdims=True)
                gradient_ema.apply(langevin_result_syn[cup_id][5])
                assert not np.any(np.isnan(syn_w[cup_id]))
                assert not np.any(np.isinf(syn_w[cup_id]))
                assert not np.any(np.isnan(syn_z[cup_id]))
                assert not np.any(np.isinf(syn_z[cup_id]))
                assert not np.any(np.isnan(syn_z2[cup_id]))
                assert not np.any(np.isinf(syn_z2[cup_id]))
                assert not np.any(np.isnan(obs_z2[cup_id]))
                assert not np.any(np.isinf(obs_z2[cup_id]))
                assert not np.any(np.isnan(syn_p[cup_id]))
                assert not np.any(np.isinf(syn_p[cup_id]))

            syn_z_seq[cup_id][:, langevin_step+1, :] = syn_z[cup_id]
            syn_z2_seq[cup_id][:, langevin_step+1, :] = syn_z2[cup_id]
            obs_z2_seq[cup_id][:, langevin_step+1, :] = obs_z2[cup_id]
            syn_e_seq[cup_id][:, langevin_step, 0] = syn_e[cup_id].reshape([-1])
            syn_w_seq[cup_id][:, langevin_step, :] = syn_w[cup_id][...,0]
            syn_p_seq[cup_id][:, langevin_step, 0] = syn_p[cup_id].reshape([-1])

        feed_dict = {model.is_training: True}
        for cup_id in cup_id_list:
            feed_dict[model.obs_z[cup_id]] = obs_z[cup_id]
            feed_dict[model.inp_z[cup_id]] = syn_z[cup_id]
            feed_dict[model.obs_z2[cup_id]] = obs_z2[cup_id]
            feed_dict[model.inp_z2[cup_id]] = syn_z2[cup_id]
        syn_ewp, obs_ewp, obs_z2, loss, _ = sess.run([model.inp_ewp, model.obs_ewp, model.obs_z2_update, model.descriptor_loss, model.des_train], feed_dict=feed_dict)
        for cup_id in cup_id_list:
            syn_e_seq[cup_id][:, -1, 0] = syn_ewp[cup_id][0].reshape([-1])
            syn_w_seq[cup_id][:, -1, :] = syn_ewp[cup_id][1][...,0]
            syn_p_seq[cup_id][:, -1, 0] = syn_ewp[cup_id][2].reshape([-1])
            obs_z2s[cup_id][idxs] = obs_z2[cup_id]

        # compute obs_w img and syn_w img if weight is situation invariant


        if flags.tb_render and ((not flags.debug and batch_id % 20 == 0) or (flags.debug and epoch % 10 == 9)):
            feed_dict = {}
            for cup_id in cup_id_list:
                feed_dict[model.summ_obs_e[cup_id]] = np.mean(obs_ewp[cup_id][0])
                feed_dict[model.summ_ini_e[cup_id]] = np.mean(syn_e_seq[cup_id][:,0])
                feed_dict[model.summ_syn_e[cup_id]] = np.mean(syn_ewp[cup_id][0])
                feed_dict[model.summ_obs_p[cup_id]] = np.mean(obs_ewp[cup_id][2])
                feed_dict[model.summ_ini_p[cup_id]] = np.mean(syn_p_seq[cup_id][:,0])
                feed_dict[model.summ_syn_p[cup_id]] = np.mean(syn_ewp[cup_id][2])
                feed_dict[model.summ_descriptor_loss[cup_id]] = loss[cup_id]
                feed_dict[model.summ_obs_bw[cup_id]] = obs_bw_img
                feed_dict[model.summ_obs_fw[cup_id]] = obs_fw_img
                feed_dict[model.summ_syn_bw[cup_id]] = syn_bw_img
                feed_dict[model.summ_syn_fw[cup_id]] = syn_fw_img
                feed_dict[model.summ_obs_im[cup_id]] = vu.visualize(cup_id, obs_z)
                feed_dict[model.summ_syn_im[cup_id]] = vu.visualize(cup_id, syn_z)
                feed_dict[model.summ_obs_w[cup_id]] = obs_ewp[cup_id][1][-1,:,0]
                feed_dict[model.summ_syn_w[cup_id]] = syn_ewp[cup_id][1][-1,:,0]
                feed_dict[model.summ_syn_e_im[cup_id]] = vu.plot_e(syn_e_seq[cup_id], obs_ewp[cup_id][0])
                feed_dict[model.summ_syn_p_im[cup_id]] = vu.plot_e(syn_p_seq[cup_id], obs_ewp[cup_id][2])
                feed_dict[model.summ_g_avg[cup_id]] = g_avg[cup_id]

            summ = sess.run(model.all_summ, feed_dict=feed_dict)
        else:
            feed_dict = {}
            for cup_id in cup_id_list:
                feed_dict[model.summ_obs_e[cup_id]] = np.mean(obs_ewp[cup_id][0])
                feed_dict[model.summ_ini_e[cup_id]] = np.mean(syn_e_seq[cup_id][:,0])
                feed_dict[model.summ_syn_e[cup_id]] = np.mean(syn_ewp[cup_id][0])
                feed_dict[model.summ_obs_p[cup_id]] = np.mean(obs_ewp[cup_id][2])
                feed_dict[model.summ_ini_p[cup_id]] = np.mean(syn_p_seq[cup_id][:,0])
                feed_dict[model.summ_syn_p[cup_id]] = np.mean(syn_ewp[cup_id][2])
                feed_dict[model.summ_descriptor_loss[cup_id]] = loss[cup_id]
                feed_dict[model.summ_obs_w[cup_id]] = obs_ewp[cup_id][1][-1,:,0]
                feed_dict[model.summ_syn_w[cup_id]] = syn_ewp[cup_id][1][-1,:,0]
                feed_dict[model.summ_g_avg[cup_id]] = g_avg[cup_id]

            summ = sess.run(model.scalar_summ, feed_dict=feed_dict)

        train_writer.add_summary(summ, global_step=epoch * batch_num + batch_id)

        print('\rE%dB%d/%d(C%d): Obs.E: %f, Obs.P: %f, Ini.E: %f, Ini.P: %f, Syn.E: %f, Syn.P: %f, Loss: %f, Time: %f'%(
            epoch, batch_id, batch_num, cup_id, 
            np.mean([np.mean(obs_ewp[cup_id][0]) for cup_id in cup_id_list]), 
            np.mean([np.mean(obs_ewp[cup_id][2]) for cup_id in cup_id_list]),
            np.mean([np.mean(syn_e_seq[cup_id][0]) for cup_id in cup_id_list]), 
            np.mean([np.mean(syn_p_seq[cup_id][0]) for cup_id in cup_id_list]),
            np.mean([np.mean(syn_ewp[cup_id][0]) for cup_id in cup_id_list]), 
            np.mean([np.mean(syn_ewp[cup_id][2]) for cup_id in cup_id_list]),
            np.mean(loss.values()), time.time() - t0), end='')
        
    data = {
        'cup_id': cup_id, 
        'obs_z' : obs_z, 
        'obs_z2' : obs_z2, 
        'obs_ewp' : obs_ewp,
        'syn_e' : syn_e_seq, 
        'syn_z' : syn_z_seq, 
        'syn_z2' : syn_z2_seq, 
        'syn_w' : syn_w_seq,
        'syn_p' : syn_p_seq,
        'g_avg' : g_avg
    }

    pickle.dump(data, open(os.path.join(fig_dir, '%04d.pkl'%(epoch)), 'wb'))
    saver.save(sess, os.path.join(model_dir, '%04d.ckpt'%(epoch)))

    pickle.dump(obs_z2s, open(os.path.join(fig_dir, '%04d.obs_z2s.pkl'%epoch), 'wb'))
    print()

# Note: z_size=2 doesn't work. data is too distorted to show real geometrical relationship
# python3 train.py --batch_size 8 --step_size 0.05 --adaptive_langevin --langevin_steps 90 --clip_norm_langevin --situation_invariant --name situation_invariant --d_lr 1e-4 --penalty_strength 1e-2
