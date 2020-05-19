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

epochs = [145, 144, 144, 145, 57, 83, 57]

for exp in [5]:
    name = 'exp%d'%exp
    epoch = epochs[exp]
    data = pickle.load(open(os.path.join('../logs/figs/logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, GC, syn_hand, SC, obs_hand, OC, GE, SE, OE, g_ema, obs_z = data
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
    for i in range(len(syn_hand)):
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
            visualizer.visualize_weight(obs_hand[i], OC[i,:,0], '%s-%d/%d-dem-%f'%(name, epoch, i, OE[i]))
            visualizer.visualize_weight(gen_hand[i], GC[i,:,0], '%s-%d/%d-gen-%f'%(name, epoch, i, GE[i]))
            visualizer.visualize_weight(syn_hand[i], SC[i,:,0], '%s-%d/%d-syn-%f'%(name, epoch, i, SE[i]))
        except:
            pass