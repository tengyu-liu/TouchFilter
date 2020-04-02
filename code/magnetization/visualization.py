import os
import pickle
import numpy as np
import sys

import mayavi.mlab as mlab
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.vis_util import VisUtil

from sklearn.manifold import TSNE

visualizer = VisUtil()

for idx_A in [1]: #range(9):
    for idx_B in [3]: #range(idx_A+1, 9):
        for dist_mult in [10]:
            os.makedirs('figs/mag/%d_%d_%f'%(idx_A, idx_B, dist_mult), exist_ok=True)
            data = pickle.load(open('mag_result_%d_%d_%f.pkl'%(idx_A, idx_B, dist_mult), 'rb'))

            # z2_manifold = TSNE().fit_transform(data['z2'].reshape([-1, 10])).reshape([data['z2'].shape[0], data['z2'].shape[1], 2])

            for i_batch in range(data['z'].shape[0]):
                for i_step in [0, 1000]: #range(0, data['z'].shape[1]):
                    print('V', idx_A, idx_B, i_batch, i_step)
                    # distance to src and tgt
                    plt.clf()
                    plt.plot(data['d_src'][i_batch, :i_step, 0], label='distance to source')
                    plt.plot(data['d_tgt'][i_batch, :i_step, 0], label='distance to target')
                    plt.xlim(0, data['z'].shape[1])
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.savefig('figs/mag/%d_%d_%f/dist_%03d_%03d.png'%(idx_A, idx_B, dist_mult, i_batch, i_step))
                    # grasping itself
                    mlab.clf()
                    visualizer.visualize(3, data['z'][i_batch, i_step])
                    mlab.savefig('figs/mag/%d_%d_%f/vis_%03d_%03d.png'%(idx_A, idx_B, dist_mult, i_batch, i_step))
                    # z2 manifold plot
                    # plt.clf()
                    # plt.scatter(z2_manifold[...,0].reshape([-1]), z2_manifold[...,1].reshape([-1]), s=1, c=data['e'].reshape([-1]))
                    # plt.plot(z2_manifold[i_batch,:,0], z2_manifold[i_batch,:,1],c='red')
                    # plt.scatter(z2_manifold[i_batch, i_step, 0], z2_manifold[i_batch, i_step, 1], s=20, c='red')
                    # plt.axis('off')
                    # plt.savefig('figs/mag/%d_%d_%f/manifold_%03d_%03d.png'%(idx_A, idx_B, dist_mult, i_batch, i_step))
        exit()