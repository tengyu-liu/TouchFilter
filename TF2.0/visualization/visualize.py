import copy
import os
import sys
import pickle

import numpy as np
import trimesh as tm
import mayavi.mlab as mlab

import mat_trans as mt
from forward_kinematics import ForwardKinematic

import matplotlib.pyplot as plt

# name = sys.argv[1]
# epoch = int(sys.argv[2])
# batch = int(sys.argv[3])

pca = pickle.load(open('../pca/pkl44/pca_2.pkl', 'rb'))
pca_components = pca.components_
pca_mean = pca.mean_
pca_var = pca.explained_variance_

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
    z_ = np.concatenate([np.matmul(hand_z[...,:-9], np.expand_dims(np.sqrt(pca_var), axis=-1) * pca_components) + pca_mean, hand_z[...,-9:]], axis=-1)
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


# data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../figs', name, '%04d-%d.pkl'%(epoch, batch)), 'rb'))

# cup_id = data['cup_id']
# cup_r = data['cup_r']
# obs_z = data['obs_z']
# syn_e_seq = data['syn_e']
# syn_z_seq = data['syn_z']

cup_id = 1
cup_r = [[[1,0,0],[0,1,0],[0,0,1]]]
syn_z = np.random.normal(size=([11]))
syn_z[-3:] = [0, 0.2, 0.2]
syn_z[-9:-3] = 0
syn_z[-9] = 1
syn_z[-6] = 1
visualize(cup_id, cup_r[0], syn_z)
mlab.show()

# for i_batch in range(len(cup_r)):
#     for i_seq in range(len(syn_z_seq)):
#         mlab.clf()
#         visualize(cup_id, cup_r[i_batch], syn_z_seq[i_seq][i_batch])
#         mlab.savefig('../figs/%s-%04d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq))
#     os.system('ffmpeg -i ../figs/%s-%04d-%d-%d-%%d.png ../figs/%s-%04d-%d-%d.gif'%(name, epoch, batch, i_batch, name, epoch, batch, i_batch))
#     for i_seq in range(len(syn_z_seq)):
#         os.remove('../figs/%s-%04d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq))
