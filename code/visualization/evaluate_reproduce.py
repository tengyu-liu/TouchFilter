import copy
import os
import sys
import pickle

import numpy as np
import trimesh as tm
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mat_trans as mt
from forward_kinematics import ForwardKinematic

import matplotlib.pyplot as plt
from pyquaternion.quaternion import Quaternion as Q

mlab.options.offscreen = True

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

ffmpeg = 'ffmpeg'
if os.name == 'nt':
    ffmpeg = 'ffmpeg4'

def rotation_matrix(rot):
    a1 = rot[:,0]
    a2 = rot[:,1]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    eye = np.eye(4)
    eye[:3,:3] = np.stack([b1, b2, b3], axis=-1)
    return eye

cup_models = {i: tm.load_mesh('../../data/cups/onepiece/%d.obj'%i) for i in range(1,9)}
obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

def visualize(cup_id, hand_z):
    cup_model = cup_models[cup_id]
    mlab.triangular_mesh(cup_model.vertices[:,0], cup_model.vertices[:,1], cup_model.vertices[:,2], cup_model.faces, color=(0, 0, 1))

    z_ = hand_z
    jrot = z_[:22]
    grot = np.reshape(z_[22:28], [3, 2])
    gpos = z_[28:]

    grot = mt.quaternion_from_matrix(rotation_matrix(grot))

    qpos = np.concatenate([gpos, grot, jrot])

    xpos, xquat = ForwardKinematic(qpos)

    obj_base = '../../data/hand'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

    for pid in range(4, 25):
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
            p.apply_translation(xpos[pid,:])
            mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
        except:
            continue

import sys
step = int(sys.argv[1])
step_size = float(sys.argv[2])

reproduce = 'reproduce[%dx%f]'%(step, step_size)

train_data = pickle.load(open('../figs/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
local_data = pickle.load(open('../%s.pkl'%reproduce, 'rb'))

# print('Z')
# for i in range(16):
#     for j in range(91):
#         print('%.2f '%(np.linalg.norm(train_data['syn_z'][i,j,:] - local_data['z'][i,j,:]) / np.linalg.norm(train_data['syn_z'][i,j,:])), end='')
#     print()
# print()

# print('Z2')
# for i in range(16):
#     for j in range(91):
#         print('%.2f '%(np.linalg.norm(train_data['syn_z2'][i,j,:] - local_data['z2'][i,j,:]) / np.linalg.norm(train_data['syn_z2'][i,j,:])), end='')
#     print()

os.makedirs('figs', exist_ok=True)
os.makedirs('figs/%s'%reproduce, exist_ok=True)

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,480))

for i in range(16):
    mlab.clf()
    visualize(3, train_data['syn_z'][i,-1,:])
    mlab.savefig('figs/%s/train_%d.png'%(reproduce, i))
    mlab.clf()
    visualize(3, local_data['z'][i,-1,:])
    mlab.savefig('figs/%s/local_%d.png'%(reproduce, i))
    