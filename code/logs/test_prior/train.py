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

if flags.tb_render:
    from utils.vis_util import VisUtil
    # load vis_util
    vu = VisUtil()


print('name', flags.name)
print('restore_epoch', flags.restore_epoch)
print('restore_batch', flags.restore_batch)
print('epochs', flags.epochs)
print('batch_size', flags.batch_size)
print('langevin_steps', flags.langevin_steps)
print('step_size', flags.step_size)
print('situation_invariant', flags.situation_invariant)
print('adaptive_langevin', flags.adaptive_langevin)
print('clip_norm_langevin', flags.clip_norm_langevin)
print('debug', flags.debug)
print('d_lr', flags.d_lr)
print('beta1', flags.beta1)
print('beta2', flags.beta2)
print('tb_render', flags.tb_render)

f = open('history.txt', 'a')
f.write('[%s] python %s\n'%(str(datetime.datetime.now()), ' '.join(sys.argv)))
f.close()

project_root = os.path.join(os.path.dirname(__file__), '..')

# load obj
cup_id_list = [1,2,3,5,6,7,8]
if flags.debug:
    cup_id_list = [1]

cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

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
                cup_rotation = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotation_matrix
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
                hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                cup_rs[cup_id].append(cup_rotation)
                obs_zs[cup_id].append(hand_z)
                if flags.debug:
                    break
            if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
                break
        if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
            break

cup_rs = {i:np.array(x) for (i,x) in cup_rs.items()}
obs_zs = {i:np.array(x) for (i,x) in obs_zs.items()}
all_zs = np.array(all_zs)

stddev = np.std(all_zs, axis=0, keepdims=True)
mean = np.mean(all_zs, axis=0, keepdims=True)
z_min = np.min(all_zs, axis=0, keepdims=True)
z_max = np.max(all_zs, axis=0, keepdims=True)

print('Z-std', stddev)

minimum_data_length = min(len(cup_rs[cup_id]) for cup_id in cup_id_list)
data_idxs = {cup_id: np.arange(len(cup_rs[cup_id])) for cup_id in cup_id_list}
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
        cup_id = cup_id_list[batch_id % len(cup_id_list)]
        item_id = batch_id // len(cup_id_list)
        idxs = shuffled_idxs[cup_id][flags.batch_size * item_id : flags.batch_size * (item_id + 1)]

        cup_r = cup_rs[cup_id][idxs]
        obs_z = obs_zs[cup_id][idxs]
        syn_z = np.zeros(obs_z.shape)
        syn_z[:,:22] = 0
        syn_z[:,-9:] = obs_z[:,-9:]
        syn_z[:,-3:] += np.random.normal(scale=flags.random_scale, size=(flags.batch_size, 3))

        syn_z_seq = [syn_z]
        syn_e_seq = []
        syn_w_seq = []
        syn_p_seq = []

        # ini_e = sess.run(model.inp_e[cup_id], feed_dict={model.cup_r: cup_r, model.inp_z: ini_z})

        update_mask = np.ones(syn_z.shape)
        update_mask[:,-9:-3] = 0.0    # We disallow grot update

        for langevin_step in range(flags.langevin_steps):
            syn_z, syn_e, syn_w, syn_p, g_avg = sess.run(model.syn_zewpg[cup_id], feed_dict={model.cup_r: cup_r, model.inp_z: syn_z, model.update_mask: update_mask})
            # print(langevin_step, syn_z, syn_e, np.any(np.isnan(syn_w)), np.any(np.isinf(syn_w)))
            syn_z[:,:22] = np.clip(syn_z[:,:22], z_min[:,:22], z_max[:,:22])
            assert not np.any(np.isnan(syn_w)) 
            assert not np.any(np.isinf(syn_w)) 
            assert not np.any(np.isnan(syn_z)) 
            assert not np.any(np.isinf(syn_z)) 

            syn_z_seq.append(copy.deepcopy(syn_z))
            syn_e_seq.append(copy.deepcopy(syn_e))
            syn_w_seq.append(copy.deepcopy(syn_w))
            syn_p_seq.append(copy.deepcopy(syn_p))
            # if langevin_step % 100 == 99:
            #     syn_ew, obs_ew, loss, _ = sess.run([model.inp_ew[cup_id], model.obs_ew[cup_id], model.descriptor_loss[cup_id], model.des_train[cup_id]], feed_dict={
            #         model.cup_r: cup_r, model.obs_z: obs_z, model.inp_z: syn_z
            #     })

        syn_ewp, obs_ewp, loss, _ = sess.run([model.inp_ewp[cup_id], model.obs_ewp[cup_id], model.descriptor_loss[cup_id], model.des_train[cup_id]], feed_dict={
            model.cup_r: cup_r, model.obs_z: obs_z, model.inp_z: syn_z
        })
        syn_e_seq.append(copy.deepcopy(syn_ewp[0]))
        syn_w_seq.append(copy.deepcopy(syn_ewp[1]))
        syn_p_seq.append(copy.deepcopy(syn_ewp[2]))

        # compute obs_w img and syn_w img if weight is situation invariant

        if flags.tb_render and ((not flags.debug and batch_id % 20 == 0) or (flags.debug and epoch % 10 == 9)):
            obs_im = vu.visualize(cup_id, cup_r, obs_z)
            syn_im = vu.visualize(cup_id, cup_r, syn_z)
            syn_e_im = vu.plot_e(syn_e_seq, obs_ewp[0])
            syn_p_im = vu.plot_e(syn_p_seq, obs_ewp[2])

            syn_bw_img, syn_fw_img = vu.visualize_hand(syn_ewp[1])
            obs_bw_img, obs_fw_img = vu.visualize_hand(obs_ewp[1])
            summ = sess.run(model.all_summ, feed_dict={
                model.summ_obs_e: obs_ewp[0], 
                model.summ_ini_e: syn_e_seq[0], 
                model.summ_syn_e: syn_ewp[0], 
                model.summ_obs_p: obs_ewp[2], 
                model.summ_ini_p: syn_p_seq[0], 
                model.summ_syn_p: syn_ewp[2], 
                model.summ_descriptor_loss: loss,
                model.summ_obs_bw: obs_bw_img, 
                model.summ_obs_fw: obs_fw_img, 
                model.summ_syn_bw: syn_bw_img,
                model.summ_syn_fw: syn_fw_img,
                model.summ_obs_im: obs_im,
                model.summ_syn_im: syn_im, 
                model.summ_obs_w: obs_ewp[1][-1,:,0],
                model.summ_syn_w: syn_ewp[1][-1,:,0],
                model.summ_syn_e_im: syn_e_im,
                model.summ_syn_p_im: syn_p_im,
                model.summ_g_avg : g_avg
            })
        else:
            summ = sess.run(model.scalar_summ, feed_dict={
                model.summ_obs_e: obs_ewp[0], 
                model.summ_ini_e: syn_e_seq[0], 
                model.summ_syn_e: syn_ewp[0], 
                model.summ_obs_p: obs_ewp[2], 
                model.summ_ini_p: syn_p_seq[0], 
                model.summ_syn_p: syn_ewp[2], 
                model.summ_descriptor_loss: loss,
                model.summ_obs_w: obs_ewp[1][-1,:,0],
                model.summ_syn_w: syn_ewp[1][-1,:,0],
                model.summ_g_avg : g_avg
            })

        train_writer.add_summary(summ, global_step=epoch * batch_num + batch_id)

        assert not (np.any(np.isnan(syn_z_seq)) or np.any(np.isinf(syn_z_seq)))
        print('\rE%dB%d/%d(C%d): Obs.E: %f, Obs.P: %f, Ini.E: %f, Ini.P: %f, Syn.E: %f, Syn.P: %f, Loss: %f, Time: %f'%(
            epoch, batch_id, batch_num, cup_id, 
            obs_ewp[0], obs_ewp[2],
            syn_e_seq[0], syn_p_seq[0],
            syn_ewp[0], syn_ewp[2],
            loss, time.time() - t0), end='')
        
        if item_id % 100 == 0:
            data = {
                'cup_id': cup_id, 
                'cup_r' : cup_r, 
                'obs_z' : obs_z, 
                'obs_e' : obs_ewp[0], 
                'obs_w' : obs_ewp[1], 
                'obs_p' : obs_ewp[2],
                'syn_e' : syn_e_seq, 
                'syn_z' : syn_z_seq, 
                'syn_w' : syn_w_seq,
                'syn_p' : syn_p_seq,
            }

            pickle.dump(data, open(os.path.join(fig_dir, '%04d-%d.pkl'%(epoch, batch_id)), 'wb'))
            saver.save(sess, os.path.join(model_dir, '%04d-%d.ckpt'%(epoch, batch_id)))

    print()

# Note: z_size=2 doesn't work. data is too distorted to show real geometrical relationship
# python3 train.py --batch_size 8 --step_size 0.05 --adaptive_langevin --langevin_steps 90 --clip_norm_langevin --situation_invariant --name situation_invariant --d_lr 1e-4 --penalty_strength 1e-2
