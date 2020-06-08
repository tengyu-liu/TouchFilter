import os
import pickle
import time

import numpy as np

from Simulator import Simulator

energies = []
distances = []

for epoch in range(200):
    try:
        os.system('scp antelope:/home/tengyu/github/TouchFilter/code/figs/nn_prior_lr-5/%04d-300.pkl ../figs/nn_prior_lr-5/'%epoch)
        data = pickle.load(open('../figs/nn_prior_lr-5/%04d-300.pkl'%epoch, 'rb'))
    except:
        break
    
    syn_z = data['syn_z'][:,-1,:]

    es = data['syn_e'][:,-1,:]
    xs = []

    sim = Simulator()
    sim.connect(with_gui=True)

    def trial(i):
        delta_xs = []
        for g in [[0,-10,0]]:
            sim.reset()
            sim.generate_world(syn_z[i])
            sim.set_gravity(g)
            for j in range(1000):
                sim.simulate()
                if np.linalg.norm(sim.get_cup_position_orientation()[0]) > 1:
                    return np.linalg.norm(sim.get_cup_position_orientation()[0])
            delta_xs.append(np.linalg.norm(sim.get_cup_position_orientation()[0]))
        return max(delta_xs)

    for i in range(len(syn_z)):
        xs.append(trial(i))
        print('\r[%d] %d/%d: %f'%(epoch, i, len(syn_z), xs[-1]), end='', flush=True)

    sim.disconnect()

    energies.append(es)
    distances.append(xs)

pickle.dump([energies, distances], open('analysis_1g.pkl', 'wb'))

"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

energies, distances = pickle.load(open('analysis.pkl', 'rb'))
plt.subplot(221)
plt.plot([np.mean(x) for x in energies])
plt.title('mean energies')
plt.subplot(222)
plt.plot([np.std(x) for x in energies])
plt.title('std energies')
plt.subplot(223)
plt.plot([np.min(x) for x in distances])
plt.title('min distances')
plt.subplot(224)
plt.plot([np.mean(x) for x in distances])
plt.title('mean distances')
plt.show()
"""