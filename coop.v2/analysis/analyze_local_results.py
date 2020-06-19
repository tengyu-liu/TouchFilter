import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from utils.viz_util import Visualizer
visualizer = Visualizer()

name = 'exp0'
last_epoch = 293

generated_energies = []
for epoch in range(last_epoch):
    data = pickle.load(open(os.path.join('../logs/%s/%04d.pkl'%(name, epoch)), 'rb'))
    obj_id, gen_hand, GC, syn_hand, SC, obs_hand, OC, GE, SE, OE, g_ema, obs_z2s, obj_rot, obj_trans = data
    generated_energies.append(np.mean(np.sum(obs_hand[:,:22] * obs_hand[:,:22], axis=-1)))

plt.plot(generated_energies)
plt.show()