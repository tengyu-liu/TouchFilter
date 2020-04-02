import copy
import os
import pickle
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from viz_util import Visualizer

data = pickle.load(open(os.path.join(os.path.dirname(__file__), 'synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl'), 'rb'))
keep_ids = (data['syn_e'] < 1.5).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
# data['syn_z2'] /= np.linalg.norm(data['syn_z2'], axis=-1, keepdims=True)
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]

mean = data['syn_w'].mean(0,keepdims=True)
std = data['syn_w'].std(0,keepdims=True)

np.save('w_mean.npy', mean)
np.save('w_std.npy', std)

pca = PCA(n_components=2)
z2_manifold = pca.fit_transform(data['syn_z2'])
knn = KDTree(z2_manifold)

np.save('../z2_analysis/pca_components.npy', pca.components_)
exit()
# for randomly sample clusters
# for each cluster, show exponential-weighted distance hands
ax = plt.subplot2grid((5, 5), (0, 0), rowspan=10, colspan=4)
ax.scatter(z2_manifold[:,0], z2_manifold[:,1], s=1, c='blue')
ax.axis('off')

colors = ['red', 'green', 'brown', 'purple', 'cyan']
v = Visualizer()
for i_cluster in range(5):
    intensities = []
    i_pt = random.randint(0, len(data['syn_z2'])-1)
    z2 = z2_manifold[i_pt]
    neighbor_distance, neighbor_indices = knn.query([z2], 10, return_distance=True, sort_results=True)
    neighbor_distance, neighbor_indices = neighbor_distance[0], neighbor_indices[0]
    ax.scatter(z2_manifold[neighbor_indices,0], z2_manifold[neighbor_indices,1], s=5, c=colors[i_cluster])
    ws = data['syn_w'][neighbor_indices]
    ws = (ws - mean) / std
    neighbor_distance += neighbor_distance[1]
    weights = 1. / (neighbor_distance * neighbor_distance)
    weights /= weights.sum()
    # ws = (ws * np.expand_dims(weights, axis=1)).sum(0)
    ws = ws.mean(0)
    ax1 = plt.subplot2grid((5,5), (i_cluster,4), projection='3d')
    v.visualize_weight(ws ** 2, ax=ax1, c=colors[i_cluster])

plt.show()
