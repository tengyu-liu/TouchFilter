import copy
import datetime
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import imageio
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion.quaternion import Quaternion as Q
from scipy.interpolate import griddata

from config import flags
from model_eval import Model
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
            #
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


flags.name = 'redo_experiment,noise_level=0.1std,continue_random_init=0.1'
flags.restore_epoch = 6
flags.restore_batch = 700
flags.langevin_steps = 90
flags.step_size = 0.1
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

N = 1000
collect_data = True
project_root = '..'
cup_id_list = [1,2,3,5,6,7,8]
cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}
#
all_zs = []
cup_rs = defaultdict(list)
obs_zs = defaultdict(list)
#
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
#
stddev = np.std(all_zs, axis=0, keepdims=True) * 10
mean = np.mean(all_zs, axis=0, keepdims=True)
z_min = np.min(all_zs, axis=0, keepdims=True)
z_max = np.max(all_zs, axis=0, keepdims=True)
# load data
if collect_data:
    # Create model
    model = Model(flags, mean, stddev, cup_id_list)
    #
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    #
    # os.makedirs(os.path.join(os.path.dirname(__file__), 'evaluate'), exist_ok=True)
    # os.makedirs(os.path.join(os.path.dirname(__file__), 'evaluate', flags.name), exist_ok=True)
    #
    # model_dir = os.path.join(os.path.dirname(__file__), 'models', flags.name)
    # fig_dir = os.path.join(os.path.dirname(__file__), 'evaluate', flags.name)
    #
    saver = tf.train.Saver(max_to_keep=0)
    if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
        saver.restore(sess, os.path.join('models', flags.name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))
    #
    count = 0
    #
    cup_id = 1
    cup_r = np.expand_dims(Q.random().rotation_matrix, axis=0)
    zs = []
    es = []
    ws = []
    #
    for i in range(N):
        #
        print('\rGenerating item %d'%i, end='', flush=True)
        #
        hand_z = np.zeros([1,31])
        hand_z[0,:-9] = np.random.random([22]) * (z_max[0,:22] - z_min[0,:22]) + z_min[0,:22]
        hand_z[0,-9:-3] = Q.random().rotation_matrix[:,:2].reshape([6])
        hand_z[0,-3:] = np.random.normal(scale=0.1, size=[3])
        #
        update_mask = np.ones(hand_z.shape)
        update_mask[:,-9:-3] = 0.0    # We disallow grot update
        #
        zs.append(hand_z[0])
        for langevin_step in range(flags.langevin_steps):
            zs.append(hand_z[0])
            hand_z, syn_e, syn_w, syn_p, g_avg = sess.run(model.syn_zewpg[cup_id], feed_dict={model.cup_r: cup_r, model.inp_z: hand_z, model.update_mask: update_mask})
            hand_z[:,:22] = np.clip(hand_z[:,:22], z_min[:,:22], z_max[:,:22])
            es.append(syn_e)
            ws.append(syn_w[0,:,0])
        zs = zs[:-1]
    #
    pickle.dump({'cup_id': cup_id, 'cup_r': cup_r, 'zs': zs, 'es': es, 'ws': ws}, open('landscape_items_with_weights.pkl', 'wb'))
    #
else:
    items = pickle.load(open('landscape_items_with_weights.pkl', 'rb'))
    #
    zs = items['zs']
    es = items['es']
    ws = items['ws']

from sklearn.manifold import TSNE

zs = np.array(zs)
es = np.array(es)
ws = np.array(ws)

tsne = TSNE(verbose=1)
embed = tsne.fit_transform(zs[:,:22])

N_grid = 1000j
xs = embed[:,0]
ys = embed[:,1]
x_min, y_min = np.min(embed, axis=0)
x_max, y_max = np.max(embed, axis=0)
grid_x, grid_y = np.mgrid[x_min:x_max:N_grid, y_min:y_max:N_grid]

grid_z2 = griddata(embed, np.log(es), (grid_x, grid_y), method='linear')

i = np.random.randint(0, N-1)

def transform_x(x):
    # return x
    return ((x - x_min) / (x_max - x_min)) * np.abs(N_grid)

def transform_y(y):
    # return y
    return ((y - y_min) / (y_max - y_min)) * np.abs(N_grid)

# fig = plt.Figure(figsize=(6.40, 4.80), dpi=100)
# ax = fig.add_subplot(111)
plt.clf()
im = plt.imshow(grid_z2)
# for i in range(1000):
#     _ = plt.plot(transform_y(embed[i * 90 : (i+1) * 90, 1]), transform_x(embed[i * 90 : (i+1) * 90, 0]), c='black', linewidth=0.5)
#     _ = plt.scatter(transform_y(embed[i * 90 + 89, 1]), transform_x(embed[i * 90 + 89, 0]), c='red', s=2)

plt.scatter(transform_y(embed[:,1]), transform_x(embed[:,0]), c='red', s=1)

plt.axis('off')
plt.colorbar(im)
# plt.show()
plt.savefig('energy_landscape.png')

plt.clf()

# Save hand pose, filter activation, signed distances, and grasping figures
parts = ['palm', 'thumb0', 'thumb1', 'thumb2', 'thumb3', 'index0', 'index1', 'index2', 'index3', 'middle0', 'middle1', 'middle2', 'middle3', 'ring0', 'ring1', 'ring2', 'ring3', 'pinky0', 'pinky1', 'pinky2', 'pinky3']
stl_dict = {obj: np.load(os.path.join(project_root, 'data/%s.sample_points.npy'%obj)) for obj in parts}

vu = VisUtil(offscreen=True)
for j in range(9):
    idx = i * 90 + j * 10 + 9
    hand_z = zs[[idx],:]
    # Hand Pose
    img1 = vu.visualize(cup_id, cup_r, hand_z)
    # Filter Visualization
    # fig = plt.Figure(figsize=(6.40, 4.80), dpi=100)
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.subplot(111, projection='3d')
    # computer hand pts, weights, distances
    xpos, xquat = vu.get_xpos_xquat(hand_z[0])
    start = 0
    end = 0
    for pid in range(4, 25):
        if '0' in parts[pid-4]: 
            continue
        p = np.matmul(Q(xquat[pid]).rotation_matrix, stl_dict[parts[pid-4]].T).T + xpos[pid]
        end += len(p)
        weight = ws[idx, start:end]
        start += len(p)
        ax.scatter(p[:,0], p[:,1], p[:,2], s=1, c=weight, cmap=cm.coolwarm)
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    ax.set_zlim([-0.3, 0.3])
    plt.show()

