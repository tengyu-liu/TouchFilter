import copy
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import skimage.io as sio
import trimesh as tm
from plotly.subplots import make_subplots
from pyquaternion.quaternion import Quaternion as Q
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

import mat_trans as mt
from forward_kinematics import ForwardKinematic

data = pickle.load(open(os.path.join(os.path.dirname(__file__), 'synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl'), 'rb'))
keep_ids = (data['syn_e'] < 1.5).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
# data['syn_z2'] /= np.linalg.norm(data['syn_z2'], axis=-1, keepdims=True)
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]

pca = PCA(n_components=2)
z2_manifold = pca.fit_transform(data['syn_z2'])
knn = KDTree(z2_manifold)

i = random.randint(0, len(data['syn_z2'])-1)
z2 = z2_manifold[i]

neighbor_distance, neighbor_indices = knn.query([z2], 5, return_distance=True, sort_results=True)
neighbor_distance, neighbor_indices = neighbor_distance[0], neighbor_indices[0]

ax = plt.subplot2grid((5, 6), (0, 0), rowspan=10, colspan=4)
ax.scatter(z2_manifold[:,0], z2_manifold[:,1], s=1, c='blue')
ax.scatter(z2_manifold[neighbor_indices,0], z2_manifold[neighbor_indices,1], s=5, c='red')
ax.axis('off')

for i in range(5):
    ax = plt.subplot2grid((5,6), (i,4))
    try:
        ax.imshow(sio.imread('figs/plotly/%04d-0.png'%(neighbor_indices[i])))
    except:
        pass
    ax.axis('off')
    ax.set_title('d=%f'%neighbor_distance[i])
    ax = plt.subplot2grid((5,6), (i,5))
    try:
        ax.imshow(sio.imread('figs/plotly/%04d-1.png'%(neighbor_indices[i])))
    except:
        pass
    ax.axis('off')

plt.show()

# TODO: 
# for randomly sample clusters
# for each cluster, show exponential-weighted distance hands
ax = plt.subplot2grid((5, 5), (0, 0), rowspan=10, colspan=4)
ax.scatter(z2_manifold[:,0], z2_manifold[:,1], s=1, c='blue')
ax.axis('off')

colors = ['red', 'green', 'yellow', 'purple', 'cyan']

# compute weighted distance
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
obj_base = os.path.join(os.path.dirname(__file__), '../../data/hand')
__zero_jrot = np.zeros([22])
__zero_grot = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
__zero_gpos = np.zeros([3])
__zero_grot = mt.quaternion_from_matrix(rotation_matrix(__zero_grot))
__zero_qpos = np.concatenate([__zero_gpos, __zero_grot, __zero_jrot])
__zero_xpos, __zero_xquat = ForwardKinematic(__zero_qpos)
__zero_stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
for pid in range(4, 25):
    p = copy.deepcopy(__zero_stl_dict[parts[pid - 4]])
    try:
        p.apply_transform(tm.transformations.quaternion_matrix(__zero_xquat[pid,:]))
        p.apply_translation(__zero_xpos[pid,:])
        __zero_stl_dict[parts[pid - 4]] = p
    except:
        continue
cup_model = tm.load_mesh(os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/3.obj'))
x,y,z,i,j,k = [], [], [], [], [], []

i_item = 0

for i_cluster in range(5):
    intensities = []
    i_pt = random.randint(0, len(data['syn_z2'])-1)
    z2 = z2_manifold[i_pt]
    neighbor_distance, neighbor_indices = knn.query([z2], 10, return_distance=True, sort_results=True)
    neighbor_distance, neighbor_indices = neighbor_distance[0], neighbor_indices[0]
    ax.scatter(z2_manifold[neighbor_indices,0], z2_manifold[neighbor_indices,1], s=5, c=colors[i_cluster])
    zs = data['syn_z'][neighbor_indices]
    for z_ in zs:
        jrot = z_[:22]
        grot = np.reshape(z_[22:28], [3, 2])
        gpos = z_[28:]
        grot = mt.quaternion_from_matrix(rotation_matrix(grot))
        qpos = np.concatenate([gpos, grot, jrot])
        xpos, xquat = ForwardKinematic(qpos)
        obj_base = os.path.join(os.path.dirname(__file__), '../../data/hand')
        stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
        intensity = []
        count = 0
        for pid in range(4, 25):
            p = copy.deepcopy(stl_dict[parts[pid - 4]])
            try:
                p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
                p.apply_translation(xpos[pid,:])
                intensity.append(-np.power(-tm.proximity.signed_distance(cup_model, p.vertices), 1/2))
                if i_item == 0:
                    x.append(__zero_stl_dict[parts[pid - 4]].vertices[:,0])
                    y.append(__zero_stl_dict[parts[pid - 4]].vertices[:,1])
                    z.append(__zero_stl_dict[parts[pid - 4]].vertices[:,2])
                    i.append(__zero_stl_dict[parts[pid - 4]].faces[:,0] + count)
                    j.append(__zero_stl_dict[parts[pid - 4]].faces[:,1] + count)
                    k.append(__zero_stl_dict[parts[pid - 4]].faces[:,2] + count)
                    count += len(p.vertices)
            except:
                raise
        intensities.append(np.hstack(intensity))
        i_item += 1
    intensities = np.array(intensities)
    neighbor_distance += neighbor_distance[1]
    weights = 1. / (neighbor_distance * neighbor_distance)
    weights /= weights.sum()
    intensities = (intensities * np.expand_dims(weights, axis=1)).sum(0)
    x, y, z, i, j, k = map(np.hstack, [x, y, z, i, j, k])
    fig = go.Figure(data=go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensities, showscale=False))
    camera = dict(eye=dict(x=0, y=0, z=-2), up=dict(x=0, y=-1, z=0))
    fig.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
    fig.write_image(os.path.join(os.path.dirname(__file__), 'weighted-activation-%d.png'%i_cluster))
    ax1 = plt.subplot2grid((5,5), (i_cluster,4))
    try:
        ax1.imshow(sio.imread(os.path.join(os.path.dirname(__file__), 'weighted-activation-%d.png'%i_cluster)))
    except:
        pass
    ax1.axis('off')

plt.show()
