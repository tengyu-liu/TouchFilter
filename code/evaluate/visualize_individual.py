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

# name = sys.argv[1]
# epoch = int(sys.argv[2])
# batch = int(sys.argv[3])

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

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,480))

syn_dir = 'synthesis/individual_z2/'
fn = 'dynamic-z2_nobn_unitz2/0028-200.pkl'
os.makedirs('figs/%s'%(fn.split('/')[0]), exist_ok=True)
os.makedirs('figs/%s'%(fn[:-4]), exist_ok=True)

data = pickle.load(open(os.path.join(syn_dir, fn), 'rb'))

for item_id in range(len(data['syn_z'])):
    mlab.clf()
    visualize(3, data['syn_z'][item_id,:])
    mlab.savefig('figs/%s/%d.png'%(fn[:-4], item_id))

