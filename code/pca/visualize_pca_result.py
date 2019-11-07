import os
import sys
import numpy as np
import pickle
from sklearn.decomposition import PCA

import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q
from mayavi import mlab

sys.path.append(os.path.dirname(__file__))

import mat_trans as mt
from forward_kinematics import ForwardKinematic

base_path = os.path.join(os.path.dirname(__file__), '../../')

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

def visualize(hand_z, offset=0):
    z_ = pca.inverse_transform(hand_z)
    jrot = np.reshape(z_[:44], [22, 2])
    grot = np.reshape(z_[44:], [3,2])
#     grot[0,0] = 1
#     grot[1,1] = -1
    gpos = np.zeros([3])
    jrot = np.arcsin((jrot / np.linalg.norm(jrot, axis=-1, keepdims=True))[:,0])
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))
    qpos = np.concatenate([gpos, grot, jrot])
    xpos, xquat = ForwardKinematic(qpos)
    obj_base = os.path.join(base_path, 'data/hand')
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
    for pid in range(4, 25):
        p = stl_dict[parts[pid - 4]]
        p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
        p.apply_translation(xpos[pid,:])
        mlab.triangular_mesh(p.vertices[:,0] + offset, p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))


N = 28

pca = pickle.load(open('pkl50/pca_%d.pkl'%N, 'rb'))

for i in range(N):
    z = np.zeros([N])
    os.makedirs('figs50/%d'%i, exist_ok=True)
    for v in range(20):
        print(i, v)
        mlab.clf()
        z[i] = v * 0.1 - 1
        visualize(z)
        mlab.savefig('figs50/%d/%d.png'%(i, v))
    os.system('ffmpeg -y -i figs50/%d/%%d.png figs50/%d.gif'%(i, i))
