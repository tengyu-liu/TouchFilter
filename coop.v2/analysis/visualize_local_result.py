import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.viz_util import Visualizer
visualizer = Visualizer()

from sklearn.manifold import TSNE

name = 'exp5'
epochs = [23]

for epoch in epochs:
    data = pickle.load(open(os.path.join('../logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, obs_hand, GE, OE, obj_rot, obj_trans = data

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
    for i in range(len(gen_hand)):
        try:
            visualizer.visualize_weight(obj_id, obs_hand[i], obj_rot[i], obj_trans[i], None, '%s-%d/%d-dem'%(name, epoch, i))
            visualizer.visualize_weight(obj_id, gen_hand[i], obj_rot[i], obj_trans[i], None, '%s-%d/%d-gen'%(name, epoch, i))
            # visualizer.visualize_weight(obj_id, syn_hand[i], obj_rot[i], obj_trans[i], SC[i,:,0], '%s-%d/%d-syn'%(name, epoch, i))
        except:
            pass
