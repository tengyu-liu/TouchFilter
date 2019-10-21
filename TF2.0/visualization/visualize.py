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

name = sys.argv[1]
epoch = int(sys.argv[2])
batch = int(sys.argv[3])

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

def visualize_hand(fig, weights, rows, i):
    xpos, xquat = ForwardKinematic(np.zeros([53]))

    obj_base = '../../data/hand'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

    start = 0
    end = 0

    for pid in range(4, 25):
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            end += len(p.vertices)

            p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
            p.apply_translation(xpos[pid,:])

            ax = fig.add_subplot(rows, 2, i * 2 - 1)
            pts = p.vertices[:,2] > 0.001
            ax.scatter(p.vertices[pts,0], p.vertices[pts,1], c=weights[start:end, 0][pts])
            ax.axis('off')

            ax = fig.add_subplot(rows, 2, i * 2)
            pts = p.vertices[:,2] <= 0.001
            ax.scatter(p.vertices[pts,0], p.vertices[pts,1], c=weights[start:end, 0][pts])
            ax.axis('off')

            start += len(p.vertices)
        except:
            continue

data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../figs', name, '%04d-%d.pkl'%(epoch, batch)), 'rb'))

cup_id = data['cup_id']
cup_r = np.array(data['cup_r'])
obs_z = np.array(data['obs_z'])
obs_e = data['obs_e']
obs_w = np.array(data['obs_w'])
syn_e = np.array(data['syn_e'])
syn_z = np.array(data['syn_z'])
syn_w = np.array(data['syn_w'])

print('cup_r', cup_r.shape)
print('obs_z', obs_z.shape)
print('obs_w', obs_w.shape)
print('syn_e', syn_e.shape)
print('syn_z', syn_z.shape)
print('syn_w', syn_w.shape)

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,530))

for i_batch in range(len(cup_r)):
    for i_seq in range(len(syn_z)):
        # Draw 3D grasping
        mlab.clf()
        visualize(cup_id, cup_r[i_batch], syn_z[i_seq][i_batch])
        mlab.savefig('../figs/%s-%04d-%d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq, 1))
        # Draw feature selection map
        fig.clf()
        visualize_hand(fig, obs_w[i_batch], 2, 1)
        visualize_hand(fig, syn_w[i_seq, i_batch], 2, 2)
        ax = fig.add_subplot(221)
        ax.set_title('obs back')
        ax = fig.add_subplot(222)
        ax.set_title('obs front')
        ax = fig.add_subplot(223)
        ax.set_title('syn back')
        ax = fig.add_subplot(224)
        ax.set_title('syn front')

        fig.savefig('../figs/%s-%04d-%d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq, 2))
        # Merge two
        os.system('ffmpeg -i ../figs/%s-%04d-%d-%d-%d-%d.png -i ../figs/%s-%04d-%d-%d-%d-%d.png -filter_complex hstack ../figs/%s-%04d-%d-%d-%d.png'%(
            name, epoch, batch, i_batch, i_seq, 1, name, epoch, batch, i_batch, i_seq, 2, name, epoch, batch, i_batch, i_seq
        ))
        os.remove('../figs/%s-%04d-%d-%d-%d-1.png'%(name, epoch, batch, i_batch, i_seq))
        os.remove('../figs/%s-%04d-%d-%d-%d-2.png'%(name, epoch, batch, i_batch, i_seq))

    os.system('ffmpeg -i ../figs/%s-%04d-%d-%d-%%d.png -filter_complex "[0:v] palettegen" palette.png'%(name, epoch, batch, i_batch))
    os.system('ffmpeg -i ../figs/%s-%04d-%d-%d-%%d.png -i palette.png -filter_complex "[0:v][1:v] paletteuse" ../figs/%s-%04d-%d-%d.gif'%(name, epoch, batch, i_batch, name, epoch, batch, i_batch))
    os.remove('palette.png')
    
    for i_seq in range(len(syn_z)):
        os.remove('../figs/%s-%04d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq))
