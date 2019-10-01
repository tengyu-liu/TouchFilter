import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    name = sys.argv[1]
    epoch = int(sys.argv[2])

    _fig = 1
    for _ep in range(epoch + 1):
        obs = []
        ini = []
        syn = []
        for fn in os.listdir('figs/%s'%name):
            if fn[:4] == '%04d'%_ep:
                data = pickle.load(open('figs/%s/%s'%(name, fn), 'rb'))

                cup_id = data['cup_id']
                cup_r = data['cup_r']
                obs_z = data['obs_z']
                ini_z = data['ini_z']
                syn_z = data['syn_z']

                obs.append(obs_z)
                ini.append(ini_z)
                syn.append(syn_z)
        
        obs = np.concatenate(obs).reshape([-1])
        ini = np.concatenate(ini).reshape([-1])
        syn = np.concatenate(syn).reshape([-1])

        plt.subplot(epoch + 1, 3, _fig)
        plt.hist(obs)
        _fig += 1
        plt.subplot(epoch + 1, 3, _fig)
        plt.hist(ini)
        _fig += 1
        plt.subplot(epoch + 1, 3, _fig)
        plt.hist(syn)
        _fig += 1

    plt.show()