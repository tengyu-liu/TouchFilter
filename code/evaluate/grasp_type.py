import copy
import os
import pickle
import sys

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import trimesh as tm
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion.quaternion import Quaternion as Q
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import mat_trans as mt
from forward_kinematics import ForwardKinematic

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


data = pickle.load(open('synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
keep_ids = (data['syn_e'] < 3).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]


n_clusters = 2

tsne = TSNE()
kmeans = KMeans(n_clusters=n_clusters)
cluster_ids = kmeans.fit_predict(data['syn_w'])

w_2d = tsne.fit_transform(data['syn_w'])
for i in range(n_clusters):
    plt.scatter(w_2d[cluster_ids==i, 0], w_2d[cluster_ids==i, 1])

plt.show()

z2_2d = tsne.fit_transform(data['syn_z2'])

for i in range(n_clusters):
    plt.scatter(z2_2d[cluster_ids==i, 0], z2_2d[cluster_ids==i, 1])

plt.show()

for i in range(n_clusters):
    plt.subplot(2,2,i + 1)
    plt.hist(data['syn_e'][cluster_ids==i, 0])

plt.show()

for i in range(n_clusters):
    for j in np.where(cluster_ids == i)[0][:10]:
        mlab.clf()
        visualize(3, data['syn_z'][j])
        mlab.savefig('cluster_%d_%d.png'%(i, j))

# cluster by weights doesn't really work. Next step is cluster by SDF distances. 