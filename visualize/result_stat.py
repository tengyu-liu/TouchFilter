import pickle 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__ == '__main__':
    name = sys.argv[1]
    epoch = int(sys.argv[2])
    batch = int(sys.argv[3])

    data = pickle.load(open('figs/%s/%d-%d.pkl'%(name, epoch, batch), 'rb'))

    cup_id = data['cup_id']
    cup_r = data['cup_r']
    obs_z = data['obs_z']
    ini_z = data['ini_z']
    syn_z = data['syn_z']

    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.hist(obs_z[:,i])
    
    plt.show()