import sys
import datetime
import copy
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import scipy.io as sio
import tensorflow as tf
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q

from model import Model


class SimulatedAnnealing:
    def __init__(self, flags):
        # build model
        self.__build_model(flags)

    def __build_model(self, flags):
        for k, v in flags.flag_values_dict().items():
            print(k, v)

        self.batch_size = flags.batch_size

        f = open('history.txt', 'a')
        f.write('[%s] python %s\n'%(str(datetime.datetime.now()), ' '.join(sys.argv)))
        f.close()

        project_root = os.path.join(os.path.dirname(__file__), '../..')

        # load obj
        cup_id_list = [3]
        if flags.debug:
            cup_id_list = [1]

        # load data
        cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

        obs_zs = defaultdict(list)
        palm_directions = defaultdict(list)

        all_zs = []

        for i in cup_id_list:
            for j in range(1,11):
                mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
                annotation = cup_annotation['%d_%d'%(i,j)]
                for frame in range(len(mat_data)):
                    cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                    hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                    hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
                    hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                    hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                    all_zs.append(hand_z)
                for start_end in annotation:
                    start, end = [int(x) for x in start_end.split(':')]
                    for frame in range(start, end):
                        cup_id = i
                        cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                        hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                        hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
                        hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
                        hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                        palm_v = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotate([0,0,1])
                        palm_directions[cup_id].append(palm_v)
                        obs_zs[cup_id].append(hand_z)
                        if flags.debug:
                            break
                    if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
                        break
                if flags.debug and len(obs_zs[cup_id]) >= flags.batch_size:
                    break

        obs_zs = {i : np.array(x) for (i,x) in obs_zs.items()}
        palm_directions = {i : np.array(x) for (i,x) in palm_directions.items()}
        all_zs = np.array(all_zs)

        self.z_stddev = np.std(all_zs, axis=0, keepdims=True)
        self.z_mean = np.mean(all_zs, axis=0, keepdims=True)
        self.z_min = np.min(all_zs, axis=0, keepdims=True)
        self.z_max = np.max(all_zs, axis=0, keepdims=True)

        # load model
        self.model = Model(flags, self.mean, self.stddev, cup_id_list)
        print('Model loaded')

        # create session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        # load checkpoint
        saver = tf.train.Saver(max_to_keep=0)
        if flags.restore_batch >= 0 and flags.restore_epoch >= 0:
            saver.restore(self.sess, os.path.join(os.path.dirname(__file__), '../models', flags.name, '%04d-%d.ckpt'%(flags.restore_epoch, flags.restore_batch)))

    def __energy(self, cup_id, z, z2):
        energy = self.sess.run(self.model.inp_ewp[cup_id], feed_dict={self.model.inp_z: z, self.model.inp_z2: z2, self.model.is_training: False})[0]
        return energy
    
    def __perturb(self, z):
        z += (np.random.random(z.shape) - 0.5) * self.z_stddev
        z[:,:22] = np.clip(z[:,:22], self.z_min[:,:22], self.a_max[:,:22])
        return z
    
    def simulated_annealing(self, cup_id, initial_z, z2, n_steps, temperature_fn, keep_history=True):
        z = np.tile(initial_z.reshape([1, 31]), [self.batch_size, 1])
        z2 = np.tile(z2.reshape([1, 10]), [self.batch_size, 10])

        curr_energy = self.__energy(cup_id, z, z2)

        if keep_history:
            z_history = np.zeros([n_steps, self.batch_size, 31])
            e_history = np.zeros([n_steps, self.batch_size])

        for i_step in range(n_steps):
            if keep_history:
                z_history[i_step,...] = z
                e_history[i_step,...] = curr_energy
            new_z = self.__perturb(z)
            new_energy = self.__energy(cup_id, new_z, z2)
            acceptance = np.exp((curr_energy - new_energy) / temperature_fn(i_step))
            rand       = np.random.random(acceptance.shape)
            z[acceptance > rand] = new_z[acceptance > rand]
            curr_energy[acceptance > rand] = new_energy[acceptance > rand]
            print('\r[Step %d/%d] Energies: %s'%(i_step, n_steps, ' '.join(['%.2f'%x for x in curr_energy])), end='', flush=True)
        
        print('\r Done. Energies: %s'%(' '.join(['%.2f'%x for x in curr_energy]))), end='\n', flush=True)
        if keep_history:
            return z, curr_energy, z_history, e_history
        else:
            return z, curr_energy
