import os

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from pyquaternion.quaternion import Quaternion as Q

cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(os.path.dirname(__file__), '../data/cup_video_annotation.txt')).readlines()}

zs = []
for i in range(1,9):
    for j in range(1,11):
        mat_data = sio.loadmat('/home/tengyu/Documents/DeepSDF/MujocoSDF/data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j))['glove_data']
        annotation = cup_annotation['%d_%d'%(i,j)]
        for start_end in annotation:
            start, end = [int(x) for x in start_end.split(':')]
            for frame in range(start, end):
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
                hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                hand_jrot = np.stack([np.sin(hand_jrot), np.cos(hand_jrot)], axis=-1)
                hand_z = np.concatenate([hand_jrot.reshape([44]), hand_grot.reshape([6]), hand_gpos])
                zs.append(hand_z)

zs = np.array(zs)
print(zs.shape)

nc = 36

pca = PCA(n_components=nc, whiten=True)
xs = pca.fit_transform(zs)

np.save('data/pca_components.npy', pca.components_)
np.save('data/pca_mean.npy', pca.mean_)
np.save('data/pca_variance.npy', pca.explained_variance_)

print('encoding error:', np.mean(np.linalg.norm(np.matmul((zs-pca.mean_), pca.components_.T) / np.sqrt(np.expand_dims(pca.explained_variance_, axis=0)) - xs, axis=-1)))
print('decoding error:', np.mean(np.linalg.norm(np.matmul(xs, np.sqrt(np.expand_dims(pca.explained_variance_, axis=-1)) * pca.components_) + pca.mean_ - zs, axis=-1)))

import matplotlib.pyplot as plt
for i in range(nc):
    plt.subplot(6, 6, i+1)
    plt.hist(xs[:,i], bins=40)
plt.show()
