import os
import pickle
import time

import numpy as np

from Simulator import Simulator

energies = []
distances = []

epoch = 99

data = pickle.load(open('C:/Users/24jas/Desktop/TouchFilter/code/evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))

print(data['syn_z'].shape, data['syn_e'].shape)

syn_z = data['syn_z']
es = data['syn_e']
ps = []

sim = Simulator()
sim.connect(with_gui=False)

def trial(i):
    sim.reset()
    sim.generate_world(syn_z[i])
    sim.simulate()
    penetration = sim.get_penetration()
    return penetration

for i in range(len(syn_z)):
    p = trial(i)
    ps.append(p)
    print('\r[%d] %d/%d: %f'%(epoch, i, len(syn_z), ps[-1]), end='', flush=True)

sim.disconnect()

np.save('penetration.npy', ps)

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