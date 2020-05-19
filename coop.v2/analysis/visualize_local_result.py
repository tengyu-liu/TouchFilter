import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.viz_util import Visualizer
visualizer = Visualizer()

name = 'exp0'
epochs = range(1,2)

for epoch in epochs:
    data = pickle.load(open(os.path.join('../logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, GC, syn_hand, SC, obs_hand, OC, GE, SE, OE, g_ema, obs_z2s = data
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
        try:
            visualizer.visualize_weight(obs_hand[i], OC[i,:,0], '%s-%d/%d-dem'%(name, epoch, i))
            visualizer.visualize_weight(gen_hand[i], GC[i,:,0], '%s-%d/%d-gen'%(name, epoch, i))
            visualizer.visualize_weight(syn_hand[i], SC[i,:,0], '%s-%d/%d-syn'%(name, epoch, i))
        except:
            pass