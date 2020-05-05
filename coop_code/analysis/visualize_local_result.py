import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.viz_util import Visualizer
visualizer = Visualizer()

name = 'exp'
epochs = range(5,6)

name = 'exp'

for epoch in epochs:
    data = pickle.load(open(os.path.join('../logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, syn_hand, obs_hand, GE, SE, g_ema = data
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
        print(name, epoch, GE[i], SE[i])
        if os.path.exists('%s-%d/%d-gen.png'%(name, epoch, i)):
            continue
        try:
            visualizer.visualize_distance(3, gen_hand[i], '%s-%d/%d-gen'%(name, epoch, i))
            visualizer.visualize_distance(3, syn_hand[i], '%s-%d/%d-syn'%(name, epoch, i))
            visualizer.visualize_distance(3, obs_hand[i], '%s-%d/%d-dem'%(name, epoch, i))
        except:
            pass