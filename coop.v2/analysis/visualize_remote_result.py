import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.viz_util import Visualizer
visualizer = Visualizer()

# name = 'exp'
# epoch = 82

epochs = [14,14,14,14]

for exp in [0,1,2,3]:
    name = 'exp%d'%exp
    epoch = epochs[exp]
    data = pickle.load(open(os.path.join('../logs/figs/logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, GC, syn_hand, SC, obs_hand, OC, GE, SE, OE, g_ema, obs_z2s, obj_rot, obj_trans = data
    os.makedirs('%s-%d'%(name, epoch), exist_ok=True)
    diff = np.zeros([len(gen_hand), len(gen_hand)])
    for i in range(len(gen_hand)):
        for j in range(len(gen_hand)):
            diff[i,j] = np.linalg.norm(gen_hand[i] - gen_hand[j]) / np.linalg.norm(gen_hand[i])
    plt.clf()
    _ = plt.imshow(diff)
    plt.axis('off')
    plt.colorbar(_)
    # plt.show()
    plt.savefig('%s-%d/gen-diff.png'%(name, epoch))
    diff = np.zeros([len(syn_hand), len(syn_hand)])
    for i in range(len(syn_hand)):
        for j in range(len(syn_hand)):
            diff[i,j] = np.linalg.norm(syn_hand[i] - syn_hand[j]) / np.linalg.norm(syn_hand[i])
    plt.clf()
    _ = plt.imshow(diff)
    plt.axis('off')
    plt.colorbar(_)
    plt.savefig('%s-%d/syn-diff.png'%(name, epoch))
    diff = np.zeros([len(obs_hand), len(obs_hand)])
    for i in range(len(obs_hand)):
        for j in range(len(obs_hand)):
            diff[i,j] = np.linalg.norm(obs_hand[i] - obs_hand[j]) / np.linalg.norm(obs_hand[i])
    plt.clf()
    _ = plt.imshow(diff)
    plt.axis('off')
    plt.colorbar(_)
    plt.savefig('%s-%d/obs-diff.png'%(name, epoch))
    for i in range(4):
        # print(name, epoch, GE[i], SE[i])
        # print(OC.shape, GC.shape, SC.shape)
        # plt.subplot(311)
        # plt.hist(OC[i])
        # plt.subplot(312)
        # plt.hist(GC[i])
        # plt.subplot(313)
        # plt.hist(SC[i])
        # plt.show()
        try:
            visualizer.visualize_weight(obj_id, obs_hand[i], obj_rot[i], obj_trans[i], OC[i,:,0], '%s-%d/%d-dem'%(name, epoch, i))
            visualizer.visualize_weight(obj_id, gen_hand[i], obj_rot[i], obj_trans[i], GC[i,:,0], '%s-%d/%d-gen'%(name, epoch, i))
            visualizer.visualize_weight(obj_id, syn_hand[i], obj_rot[i], obj_trans[i], SC[i,:,0], '%s-%d/%d-syn'%(name, epoch, i))
        except:
            pass