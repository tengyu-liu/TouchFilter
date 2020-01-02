import copy
import os
import sys
import pickle

import numpy as np
import trimesh as tm
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

import mat_trans as mt
from forward_kinematics import ForwardKinematic

import matplotlib.pyplot as plt
from pyquaternion.quaternion import Quaternion as Q

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

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

project_root = os.path.join(os.path.dirname(__file__), '../..')
cup_model = tm.load_mesh('../../data/cups/onepiece/%d.obj'%3)
obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

obs_zs = []

for j in range(1,11):
    mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(3,3,j)))['glove_data']
    annotation = cup_annotation['%d_%d'%(3,j)]
    for start_end in annotation:
        start, end = [int(x) for x in start_end.split(':')]
        for frame in range(start, end):
            cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
            hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
            hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
            hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
            hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
            obs_zs.append(hand_z)

obs_zs = np.array(obs_zs)

idx = np.random.randint(0, len(obs_zs)-1)
obs_z = obs_zs[idx]
mlab.clf()
mlab.triangular_mesh(cup_model.vertices[:,0], cup_model.vertices[:,1], cup_model.vertices[:,2], cup_model.faces, color=(0, 0, 1))
jrot = obs_z[:22] * 0
grot = np.reshape(obs_z[22:28], [3, 2])
gpos = obs_z[28:]

grot = mt.quaternion_from_matrix(rotation_matrix(grot))

qpos = np.concatenate([gpos, grot, jrot])

xpos, xquat = ForwardKinematic(qpos)

obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

for pid in range(4, 25):
    p = copy.deepcopy(stl_dict[parts[pid - 4]])
    if parts[pid-4] == 'palm':
        q1,q2,q3 = Q(xquat[pid]).rotate([0,0,1])
        mlab.quiver3d(xpos[pid,0], xpos[pid,1], xpos[pid,2], q1, q2, q3, color=(0.0,1.0,0.0))
    try:
        p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
        p.apply_translation(xpos[pid,:])
        mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
    except:
        continue

mlab.show()

