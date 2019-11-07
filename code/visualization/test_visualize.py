import copy
import os
import sys
import pickle

from collections import defaultdict
import scipy.io as sio

import numpy as np
import trimesh as tm
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

cup_models = {i: tm.load_mesh('../../data/cups/onepiece/%d.obj'%i) for i in range(1,9)}
obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

def visualize(cup_id, cup_r, hand_z, offset=0):
    cup_model = cup_models[cup_id]
    cvert = np.matmul(cup_r, cup_model.vertices.T).T
    if offset == 0:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 1, 0))
    else:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 0, 1))

    z_ = hand_z
    jrot = np.reshape(z_[:44], [22, 2])
    grot = np.reshape(z_[44:50], [3, 2])
    gpos = z_[50:]

    jrot = np.arcsin((jrot / np.linalg.norm(jrot, axis=-1, keepdims=True))[:,0])
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
            mlab.triangular_mesh(p.vertices[:,0] + offset, p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
        except:
            continue

def visualize_hand(fig, weights, rows, i):
    xpos, xquat = ForwardKinematic(np.zeros([53]))

    obj_base = '../../data'
    stl_dict = {obj: np.load(os.path.join(obj_base, '%s.surface_sample.npy'%obj)) for obj in parts}

    start = 0
    end = 0

    for pid in range(4, 25):
        if '0' in parts[pid - 4]:
            continue
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            end += len(p)

            p = np.matmul(Q().rotation_matrix, p.T).T
            p += xpos[[pid], :]

            ax = fig.add_subplot(rows, 2, i * 2 - 1)
            pts = p[:,2] > 0.001
            ax.scatter(p[pts,0], p[pts,1], c=weights[start:end, 0][pts])
            ax.axis('off')

            ax = fig.add_subplot(rows, 2, i * 2)
            pts = p[:,2] <= 0.001
            ax.scatter(p[pts,0], p[pts,1], c=weights[start:end, 0][pts])
            ax.axis('off')

            start += len(p)
        except:
            raise
            continue

# load obj
cup_id_list = [1,2,3,4,5,6,7,8]

project_root = '../../'
cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

for i in cup_id_list:
    center = np.mean(cup_models[i].bounding_box.vertices, axis=0)
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
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
                hand_jrot = np.stack([np.sin(hand_jrot), np.cos(hand_jrot)], axis=-1)
                hand_z = np.concatenate([hand_jrot.reshape([44]), hand_grot.reshape([6]), hand_gpos])
                cup_rs[cup_id].append(cup_rotation)
                obs_zs[cup_id].append(hand_z)

cup_rs = {i:np.array(x) for (i,x) in cup_rs.items()}
obs_zs = {i:np.array(x) for (i,x) in obs_zs.items()}

zs = np.vstack(obs_zs.values())
stddev = np.std(zs, axis=0)

for c in range(1,9):
    for i in np.random.permutation(len(cup_rs[c])):
        visualize(c, cup_rs[c][i], obs_zs[c][i])
        visualize(c, cup_rs[c][i], obs_zs[c][i] + np.random.normal(scale=stddev, size=stddev.shape) * 0.05)
        mlab.show()