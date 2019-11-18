import random
import sys
import datetime
import copy
import os
import pickle
import time
from collections import defaultdict

import imageio
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q

from config import flags
from model import Model
from utils.vis_util import VisUtil

def save_gif(syn_zs, count):
    try:
        imgs = []
        save = str(input("Save? "))
        if save == 'y':
            for hand_z in syn_zs:
                syn_im = vu.visualize(cup_id, cup_r, hand_z)[0]
                plt.clf()
                plt.axis('off')
                plt.imshow(syn_im)
                plt.pause(1e-5)
                imgs.append(syn_im)

            a, e, d, f = mlab.view()
            for k in range(40):
                a = (a + 9) % 360
                mlab.view(a, e, d)
                syn_img = mlab.screenshot()
                plt.clf()
                plt.axis('off')
                plt.imshow(syn_im)
                plt.pause(1e-5)
                imgs.append(syn_img)
            count += 1
            imageio.mimsave(os.path.join(fig_dir, '%d.gif'%count), imgs)
        return count
    except:
        return count


flags.name = 'with_noise,0.1x90'
flags.restore_epoch = 6
flags.restore_batch = 3500
flags.langevin_steps = 90
flags.step_size = 0.01
flags.batch_size = 1
flags.adaptive_langevin = True
flags.clip_norm_langevin = True

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

# load data
vu = VisUtil(offscreen=True)
project_root = os.path.join(os.path.dirname(__file__), '..')

cup_id_list = [1,2,3,5,6,7,8]
cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

all_zs = []
cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

for i in cup_id_list:
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
        for frame in range(len(mat_data)):
            cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
            hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
            hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
            hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
            hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
            all_zs.append(hand_z)
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
all_zs = np.array(all_zs)

stddev = np.std(all_zs, axis=0, keepdims=True)
mean = np.mean(all_zs, axis=0, keepdims=True)
z_min = np.min(all_zs, axis=0, keepdims=True)
z_max = np.max(all_zs, axis=0, keepdims=True)

# Create model
model = Model(flags, mean, stddev, cup_id_list)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

os.makedirs(os.path.join(os.path.dirname(__file__), 'evaluate'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'evaluate', flags.name), exist_ok=True)

model_dir = os.path.join(os.path.dirname(__file__), 'models', flags.name)
fig_dir = os.path.join(os.path.dirname(__file__), 'evaluate', flags.name)

saver = tf.train.Saver(max_to_keep=0)
if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(model_dir, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

count = 0

plt.ion()

while True:
    # Initial condition
    cup_id = random.choice([5,7,8])
    # item_id = random.choice(list(range(len(cup_rs[cup_id]))))
    # cup_r = np.expand_dims(cup_rs[cup_id][item_id], axis=0)

    # hand_z = np.zeros([1, 31])
    # hand_z[0,-9:] = obs_zs[cup_id][item_id][-9:]
    # hand_z[0,-3:] += np.random.normal(scale=0.03, size=[3])

    cup_r = np.expand_dims(Q.random().rotation_matrix, axis=0)
    hand_z = np.zeros([1,31])
    hand_z[0,-9:-3] = Q.random().rotation_matrix[:,:2].reshape([6])
    hand_z[0,-3:] = np.random.normal(scale=0.1, size=[3])

    update_mask = np.ones(hand_z.shape)
    update_mask[:,-9:-3] = 0.0    # We disallow grot update

    print('cup_id: ', cup_id)
    print('cup_r: ', cup_r)
    print('hand_z: ', hand_z)

    syn_zs = []

    for langevin_step in range(flags.langevin_steps):
        hand_z, syn_e, syn_w, syn_p, g_avg = sess.run(model.syn_zewpg[cup_id], feed_dict={model.cup_r: cup_r, model.inp_z: hand_z, model.update_mask: update_mask})
        hand_z[:,:22] = np.clip(hand_z[:,:22], z_min[:,:22], z_max[:,:22])
        syn_zs.append(hand_z)

    img = vu.visualize(cup_id, cup_r, syn_zs[-1])[0]
    plt.clf()
    plt.axis('off')
    plt.imshow(img)
    plt.pause(1e-5)

    count = save_gif(syn_zs, count)
