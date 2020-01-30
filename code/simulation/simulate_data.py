import os
import pickle
import time

import numpy as np

from Simulator import Simulator

data = pickle.load(open('/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/code/evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
syn_z = data['syn_z']

es = data['syn_e']
xs = []

sim = Simulator()
sim.connect(with_gui=False)

def trial(i):
    delta_xs = []
    for g in [[10,0,0],[-10,0,0],[0,10,0],[0,-10,0],[0,0,10],[0,0,-10]]:
        sim.reset()
        sim.generate_world(syn_z[i])
        sim.set_gravity(g)
        for j in range(100):
            sim.simulate()
            # time.sleep(1/240.)
        delta_xs.append(np.linalg.norm(sim.get_cup_position_orientation()[0]))
    return max(delta_xs)

for i in range(len(syn_z)):
    print('\r%d/%d'%(i, len(syn_z)), end='', flush=True)
    xs.append(trial(i))

sim.disconnect()

xs = np.array(xs)
es = es.reshape([-1])

import matplotlib.pyplot as plt

ax = plt.subplot(221)
ax.set_title('Dist of X (E < 3)')
ax.hist(xs[es < 3.0], bins=100)
ax = plt.subplot(222)
ax.set_title('Dist of X (E >= 3)')
ax.hist(xs[es >= 3.0], bins=100)
ax = plt.subplot(223)
ax.set_title('Dist of E (X < 0.1)')
ax.hist(es[xs < 0.1], bins=100)
ax = plt.subplot(224)
ax.set_title('Dist of E (X >= 0.1)')
ax.hist(es[xs >= 0.1], bins=100)
plt.show()

