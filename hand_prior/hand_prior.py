import copy
import os
import sys
import pickle
import time
from collections import defaultdict

import numpy as np
import scipy.io as sio
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q
from mayavi import mlab

sys.path.append(os.path.dirname(__file__))

import mat_trans as mt
from forward_kinematics import ForwardKinematic

# load pca
pca_mean = np.load('data/pca_mean.npy')
pca_components = np.load('data/pca_components.npy')
pca_var = np.load('data/pca_variance.npy')
print('PCA loaded.')

# load obj
cup_id_list = [1,2,3,4,5,6,7,8]
cup_models = {cup_id: tm.load_mesh('data/cups/onepiece/%d.obj'%cup_id) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open('data/cup_video_annotation.txt').readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)

for i in cup_id_list:
    center = np.mean(cup_models[i].bounding_box.vertices, axis=0)
    for j in range(1,11):
        mat_data = sio.loadmat('data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j))['glove_data']
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
                # hand_z = np.matmul((hand_z-pca_mean), pca_components.T) / np.sqrt(pca_var)
                obs_zs[cup_id].append(hand_z)

obs_zs = np.vstack([np.array(x) for (i,x) in obs_zs.items()])

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
    # z_ = hand_z
    z_ = np.matmul(hand_z, np.expand_dims(np.sqrt(pca_var), axis=-1) * pca_components) + pca_mean
    jrot = np.reshape(z_[:44], [22, 2])
    grot = np.reshape(z_[44:50], [3, 2])
    gpos = z_[50:]
    jrot = np.arcsin((jrot / np.linalg.norm(jrot, axis=-1, keepdims=True))[:,0])
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))
    qpos = np.concatenate([gpos, grot, jrot])
    xpos, xquat = ForwardKinematic(qpos)
    obj_base = '/home/tengyu/Documents/DeepSDF/MujocoSDF/data/hand/my_model-20190305T235921Z-001/my_model/mesh'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
    for pid in range(4, 25):
        p = stl_dict[parts[pid - 4]]
        p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
        p.apply_translation(xpos[pid,:])
        mlab.triangular_mesh(p.vertices[:,0] + offset, p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

errs = []
for i in range(1, 44):
    pca = PCA(n_components=i, whiten=True)
    pca.fit(obs_zs[:,:44])
    pickle.dump(pca, open('hand_prior/pca_%d.pkl'%(i), 'wb'))
    err = np.mean(np.linalg.norm(pca.inverse_transform(pca.transform(obs_zs[:,:44])) - obs_zs[:,:44], axis=-1) / np.linalg.norm(obs_zs[:,:44], axis=-1))
    print(i, err)
    errs.append(err)

plt.plot(np.arange(1,44), errs)
plt.show()


# pca = PCA(n_components=20, whiten=True)
# x_transform = pca.fit_transform(obs_zs[:,:44])

# gm = GaussianMixture()
# gm.fit(x_transform)

# plt.subplot(121)
# plt.hist(x_transform.reshape([-1]), bins=100)
# plt.subplot(122)
# plt.imshow(gm.covariances_[0])
# plt.show()

# mlab.points3d(pts[:,0], pts[:,1], pts[:,2], mode='point')
# mlab.show()

# for i in range(1,21):
#     GM = GaussianMixture(n_components=i)
#     GM.fit(obs_zs)
#     pickle.dump(GM, open('data/hand_gmm/gmm%d.pkl'%i, 'wb'))

