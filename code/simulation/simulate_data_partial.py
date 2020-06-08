import os
import pickle
import sys
import time

import numpy as np

from Simulator import Simulator

# script_idx = int(sys.argv[1])

data = pickle.load(open('/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/code/evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
syn_z = data['syn_z']

es = data['syn_e']
xs = []

sim = Simulator()
sim.connect(with_gui=False)

def trial(i):
    delta_xs = []
    for g in [[0,0,-10]]:
        sim.reset()
        sim.generate_world(syn_z[i])
        sim.set_gravity(g)
        for j in range(1000):
            sim.simulate()
            if np.linalg.norm(sim.get_cup_position_orientation()[0]) > 1:
                return False
    return True

# for i in range(script_idx * 1000, min((script_idx + 1) * 1000, len(syn_z))):
for i in range(len(syn_z)):
    xs.append(trial(i))
    print('\r%d/%d: %f%%'%(i, len(syn_z), sum(xs)/len(xs)*100), end='', flush=True)

sim.disconnect()

xs = np.array(xs)
# np.save('xs_%d.npy'%script_idx, xs)
np.save('xs.npy', xs)
