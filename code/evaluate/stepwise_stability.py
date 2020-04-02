import os
import pickle
import time

import numpy as np

from Simulator import Simulator


for ss in [0.1]:
    data = pickle.load(open('stepwise/reproduce[%dx%f].pkl'%(250,ss), 'rb'))
    energies = data['e'][:,:-1,0]
    distances = np.zeros([16,250])
    for i in range(16):
        for j in range(250):
            if j % 5 != 0:
                continue
            print('%.1f %d %d'%(ss, i, j))

            sim = Simulator()
            sim.connect(with_gui=False)

            def trial(z):
                delta_xs = []
                for g in [[0,-10,0]]:
                    sim.reset()
                    sim.generate_world(z)
                    sim.set_gravity(g)
                    for j in range(1000):
                        sim.simulate()
                        if np.linalg.norm(sim.get_cup_position_orientation()[0]) > 1:
                            return np.linalg.norm(sim.get_cup_position_orientation()[0])
                    delta_xs.append(np.linalg.norm(sim.get_cup_position_orientation()[0]))
                return max(delta_xs)

            distances[i,j] = trial(data['z'][i,j])

            sim.disconnect()
    
    pickle.dump([energies, distances], open('stepwise/stability_%.1f.pkl'%ss, 'wb'))
