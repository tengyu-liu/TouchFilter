import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as sio
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

data = pickle.load(open('synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
keep_ids = (data['syn_e'] < 1.5).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
data['syn_z2'] /= np.linalg.norm(data['syn_z2'], axis=-1, keepdims=True)
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]

tsne = TSNE()
z2_manifold = tsne.fit_transform(data['syn_z2'])
knn = KDTree(data['syn_z2'])

i = random.randint(0, len(data['syn_z2'])-1)
z2 = data['syn_z2'][i]

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
