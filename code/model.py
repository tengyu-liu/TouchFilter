import os
import pickle

import numpy as np
import tensorflow as tf

from utils.CupModel import CupModel
from utils.HandModel import HandModel
from utils.TouchFilter import TouchFilter
from utils.tf_util import fully_connected
from utils.HandPriorNN import HandPriorNN

class Model:
    def __init__(self, config, mean, stddev, cup_list):
        self.build_config(config, mean, stddev, cup_list)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
    
    def build_config(self, config, mean, stddev, cup_list):
        self.z2_size = config.z2_size
        self.situation_invariant = config.situation_invariant
        self.adaptive_langevin = config.adaptive_langevin
        self.clip_norm_langevin = config.clip_norm_langevin
        self.batch_size = config.batch_size
        self.langevin_steps = config.langevin_steps
        self.step_size = config.step_size
        self.prior_weight = config.prior_weight
        self.prior_type = config.prior_type
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.random_strength = config.random_strength
        self.dynamic_z2 = config.dynamic_z2
        self.l2_reg = config.l2_reg

        self.debug = config.debug

        self.cup_list = cup_list

        self.cup_model_path = os.path.join(os.path.dirname(__file__), '../data/cups/models')
        self.cup_restore = 199

        self.graph = tf.Graph()

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.z_mean = tf.constant(mean, dtype=tf.float32)
            self.z_weight = tf.constant(stddev, dtype=tf.float32)
            self.obs_penetration_penalty = tf.constant(0, dtype=tf.float32)
            self.syn_penetration_penalty = tf.constant(1, dtype=tf.float32)

    def build_input(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.inp_z2 = {i: tf.placeholder(tf.float32, [self.batch_size, self.z2_size], 'input_latent') for i in self.cup_list}
            self.obs_z2 = {i: tf.placeholder(tf.float32, [self.batch_size, self.z2_size], 'obs_latent') for i in self.cup_list}
            self.inp_z = {i: tf.placeholder(tf.float32, [self.batch_size, 31], 'input_hand_z') for i in self.cup_list}
            self.obs_z = {i: tf.placeholder(tf.float32, [self.batch_size, 31], 'obs_z') for i in self.cup_list}
            self.g_avg = tf.placeholder(tf.float32, [31], 'g_avg')
            self.is_training = tf.placeholder(tf.bool, [], 'is_training')
            self.update_mask = tf.placeholder(tf.float32, [self.batch_size, 31], 'update_mask')
        pass

    def build_model(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.EMA = tf.train.ExponentialMovingAverage(decay=0.99)
            self.hand_model = HandModel(self.batch_size)
            self.touch_filter = TouchFilter(self.hand_model.n_surf_pts, situation_invariant=self.situation_invariant)
            if self.prior_type == 'NN':
                self.hand_prior_nn = HandPriorNN(self.l2_reg)
            self.des_optim = tf.train.GradientDescentOptimizer(self.d_lr)
            tf.get_variable_scope().reuse_variables()
            self.cup_models = {}
            self.obs_ewp = {}
            self.langevin_dynamics = {}
            self.inp_ewp = {}
            self.syn_zzewpg = {}
            self.descriptor_loss = {}
            self.gradients = {}
            self.obs_z2_update = {}

            with tf.variable_scope(tf.get_variable_scope()):
                for i_gpu, cup_id in enumerate(self.cup_list):
                    with tf.device('/gpu:%d'%i_gpu):
                        with tf.name_scope('TOWER_%d'%i_gpu) as scope:
                            self.cup_models[cup_id] = CupModel(cup_id, self.cup_restore, self.cup_model_path)
                            self.obs_ewp[cup_id] = self.descriptor(self.obs_z[cup_id], self.obs_z2[cup_id], self.cup_models[cup_id], self.obs_penetration_penalty)
                            tf.get_variable_scope().reuse_variables()
                            self.langevin_dynamics[cup_id] = self.langevin_dynamics_fn(cup_id)
                            self.inp_ewp[cup_id] = self.descriptor(self.inp_z[cup_id], self.inp_z2[cup_id], self.cup_models[cup_id], self.syn_penetration_penalty)
                            self.syn_zzewpg[cup_id] = self.langevin_dynamics[cup_id](self.inp_z[cup_id], self.inp_z2[cup_id])
                            self.descriptor_loss[cup_id] = (tf.reduce_mean(self.obs_ewp[cup_id][0]) + tf.reduce_mean(self.obs_ewp[cup_id][2]) * self.prior_weight) - (tf.reduce_mean(self.inp_ewp[cup_id][0]) + tf.reduce_mean(self.inp_ewp[cup_id][2]) * self.prior_weight)
                            self.gradients[cup_id] = self.des_optim.compute_gradients(self.descriptor_loss[cup_id], var_list=[var for var in tf.trainable_variables()])
                            self.obs_z2_update[cup_id] = self.obs_z2[cup_id] - tf.gradients(self.obs_ewp[cup_id][0] + tf.reduce_mean(tf.norm(self.obs_z2[cup_id], axis=-1)), self.obs_z2[cup_id])[0] * self.d_lr
    
    def hand_prior_physics(self, weight, surface_normals):
        """
        Hand prior model. 
        Idea: For all contact points (w > 0.5), compute the pairwise cosine similarity 
                between their surface normals. Pick the minimum of all. 
        Intuition: Ideally the best grasp comes from when two contact points are facing 
                each other, which makes the cosine of angle between surface normals -1.
        """
        mean_prior = [tf.constant(0, dtype=tf.float32) for i in range(self.batch_size)]

        def zero():
            return tf.constant(0, dtype=tf.float32)
        
        def prior(surface_normal, index):
            contact_normals = tf.boolean_mask(surface_normal, index) 
            c_normal_1 = tf.expand_dims(contact_normals, axis=0)   # 1 x N x 3
            c_normal_2 = tf.expand_dims(contact_normals, axis=1)   # N x 1 x 3
            c_cosine_similarity = tf.reduce_sum(c_normal_1 * c_normal_2, axis=-1) # N x N
            return tf.reduce_min(c_cosine_similarity)

        for batch_i in range(self.batch_size):
            index = weight[batch_i,:,0] >= 0.5
            mean_prior[batch_i] += tf.cond(tf.reduce_any(index), lambda : prior(surface_normals[batch_i], index), zero)
        return tf.stack(mean_prior)

    def descriptor(self, hand_z, z2, cup_model, penetration_penalty):
        z_ = hand_z
        jrot = z_[:,:22]
        grot = tf.reshape(z_[:,22:28], [z_.shape[0], 3, 2])
        gpos = z_[:,28:]
        surf_pts, surf_normals = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
        surf_pts = tf.concat(list(surf_pts.values()), axis=1)
        surf_normals = tf.concat(list(surf_normals.values()), axis=1)
        z2 = z2 / tf.norm(z2, axis=-1, keepdims=True)

        # touch response
        energy, weight = self.touch_filter(surf_pts, surf_normals, self.hand_model.pts_feature, z2, cup_model, penetration_penalty, self.is_training)
        
        if self.prior_type == "NN":
            hand_prior = self.hand_prior_nn(hand_z)
        elif self.prior_type == "Phys":
            hand_prior = self.hand_prior_physics(weight, surf_normals)
        else:
            raise NotImplementedError("Prior type must be \"NN\" or \"Phys\"")
        return energy, weight, hand_prior

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, z2):
            energy, weight, hand_prior = self.descriptor(z,z2,self.cup_models[cup_id], self.syn_penetration_penalty) #+ tf.reduce_mean(z[:,:self.hand_z_size] * z[:,:self.hand_z_size]) + tf.reduce_mean(z[:,self.hand_z_size:] * z[:,self.hand_z_size:])
            grad_z = tf.gradients(tf.reduce_mean(energy) + tf.reduce_mean(hand_prior * self.prior_weight), z)[0]
            gz_abs = tf.reduce_mean(tf.abs(grad_z), axis=0)
            if self.adaptive_langevin:
                grad_z = grad_z / self.g_avg
            if self.clip_norm_langevin:
                grad_z = tf.clip_by_norm(grad_z, 31, axes=-1)
            z2g = 0
            if self.dynamic_z2:
                z2g = tf.gradients(tf.reduce_mean(energy) + tf.reduce_mean(tf.norm(z2, axis=-1)), z2)[0]
                z2g /= tf.norm(z2g, axis=-1, keepdims=True)

            grad_z = grad_z * self.z_weight[0]
            z = z - self.step_size * grad_z * self.update_mask + self.step_size * tf.random.normal(z.shape, mean=0.0, stddev=self.z_weight[0]) * self.update_mask * self.random_strength
            z2 = z2 - self.step_size * z2g + self.step_size * tf.random.normal(z2.shape) * self.random_strength
            return [z, z2, energy, weight, hand_prior, gz_abs]
            
        return langevin_dynamics

    def build_train(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            average_grads = []
            for grad_and_vars in zip(*self.gradients.values()):
                grad = tf.concat([tf.expand_dims(g, axis=0) for g, _ in grad_and_vars], axis=0)
                grad = tf.reduce_mean(grad, axis=0)
                v = grad_and_vars[0][1]
                average_grads.append((grad, v))
            
            self.des_train = self.des_optim.apply_gradients(average_grads)
    
    def build_summary(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.summ_obs_e = {i : tf.placeholder(tf.float32, [], 'summ_obs_e/%d'%i) for i in self.cup_list}
            self.summ_ini_e = {i : tf.placeholder(tf.float32, [], 'summ_ini_e/%d'%i) for i in self.cup_list}
            self.summ_syn_e = {i : tf.placeholder(tf.float32, [], 'summ_syn_e/%d'%i) for i in self.cup_list}
            self.summ_obs_p = {i : tf.placeholder(tf.float32, [], 'summ_obs_p/%d'%i) for i in self.cup_list}
            self.summ_ini_p = {i : tf.placeholder(tf.float32, [], 'summ_ini_p/%d'%i) for i in self.cup_list}
            self.summ_syn_p = {i : tf.placeholder(tf.float32, [], 'summ_syn_p/%d'%i) for i in self.cup_list}
            self.summ_obs_w = {i : tf.placeholder(tf.float32, [None], 'summ_obs_w/%d'%i) for i in self.cup_list}
            self.summ_syn_w = {i : tf.placeholder(tf.float32, [None], 'summ_syn_w/%d'%i) for i in self.cup_list}
            self.summ_g_avg = {i : tf.placeholder(tf.float32, [31], 'summ_g_avg/%d'%i) for i in self.cup_list}
            self.summ_descriptor_loss = {i : tf.placeholder(tf.float32, [], 'summ_descriptor_loss/%d'%i) for i in self.cup_list}

            scalar_summs = []
            for i in self.cup_list:
                scalar_summs.append(tf.summary.scalar('energy/obs/%d'%i, self.summ_obs_e[i]))
                scalar_summs.append(tf.summary.scalar('energy/ini/%d'%i, self.summ_ini_e[i]))
                scalar_summs.append(tf.summary.scalar('energy/syn/%d'%i, self.summ_syn_e[i]))
                scalar_summs.append(tf.summary.scalar('energy/obs/%d'%i, self.summ_obs_e[i]))
                scalar_summs.append(tf.summary.scalar('prior/obs/%d'%i, self.summ_obs_p[i]))
                scalar_summs.append(tf.summary.scalar('prior/ini/%d'%i, self.summ_ini_p[i]))
                scalar_summs.append(tf.summary.scalar('prior/syn/%d'%i, self.summ_syn_p[i]))
                scalar_summs.append(tf.summary.scalar('prior/imp/%d'%i, self.summ_ini_p[i] - self.summ_syn_p[i]))
                scalar_summs.append(tf.summary.histogram('weight/obs/%d'%i, self.summ_obs_w[i]))
                scalar_summs.append(tf.summary.histogram('weight/syn/%d'%i, self.summ_syn_w[i]))
                scalar_summs.append(tf.summary.scalar('loss/%d'%i, self.summ_descriptor_loss[i]))
                
            self.summ_obs_bw = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_bw/%d'%i) for i in self.cup_list}
            self.summ_obs_fw = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_fw/%d'%i) for i in self.cup_list}
            self.summ_syn_bw = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_bw/%d'%i) for i in self.cup_list}
            self.summ_syn_fw = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_fw/%d'%i) for i in self.cup_list}
            self.summ_obs_im = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_im/%d'%i) for i in self.cup_list}
            self.summ_syn_im = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_im/%d'%i) for i in self.cup_list}
            self.summ_syn_e_im = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_e_im/%d'%i) for i in self.cup_list}
            self.summ_syn_p_im = {i : tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_p_im/%d'%i) for i in self.cup_list}
            img_summs = []
            for i in self.cup_list:
                img_summs.append(tf.summary.image('w/obs/back/%i', self.summ_obs_bw[i]))
                img_summs.append(tf.summary.image('w/obs/front/%i', self.summ_obs_fw[i]))
                img_summs.append(tf.summary.image('w/syn/back/%i', self.summ_syn_bw[i]))
                img_summs.append(tf.summary.image('w/syn/front/%i', self.summ_syn_fw[i]))
                img_summs.append(tf.summary.image('render/obs/%i', self.summ_obs_im[i]))
                img_summs.append(tf.summary.image('render/syn/%i', self.summ_syn_im[i]))
                img_summs.append(tf.summary.image('plot/syn_e/%i', self.summ_syn_e_im[i]))
                img_summs.append(tf.summary.image('plot/syn_prior/%i', self.summ_syn_p_im[i]))

            self.scalar_summ = tf.summary.merge(scalar_summs)
            self.all_summ = tf.summary.merge(img_summs + scalar_summs)
            pass
