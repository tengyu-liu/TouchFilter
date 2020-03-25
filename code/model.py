import os
import pickle

import numpy as np
import tensorflow as tf

from utils.CupModel import CupModel
from utils.HandModel import HandModel
from utils.TouchFilter import TouchFilter
from utils.tf_util import fully_connected

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
        self.z_mean = tf.constant(mean, dtype=tf.float32)
        self.z_weight = tf.constant(stddev, dtype=tf.float32)

        self.debug = config.debug

        self.cup_list = cup_list

        self.cup_model_path = os.path.join(os.path.dirname(__file__), '../data/cups/models')
        self.cup_restore = 199
    
        self.obs_penetration_penalty = tf.constant(0, dtype=tf.float32)
        self.syn_penetration_penalty = tf.constant(1, dtype=tf.float32)

    def build_input(self):
        
        self.inp_z2 = tf.placeholder(tf.float32, [self.batch_size, self.z2_size], 'input_latent')
        self.obs_z2 = tf.placeholder(tf.float32, [self.batch_size, self.z2_size], 'obs_latent')

        self.inp_z = tf.placeholder(tf.float32, [self.batch_size, 31], 'input_hand_z')
        self.obs_z = tf.placeholder(tf.float32, [self.batch_size, 31], 'obs_z')

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.update_mask = tf.placeholder(tf.float32, [self.batch_size, 31], 'update_mask')

        self.gz_mean = tf.placeholder(tf.float32, [31])
        pass

    def build_model(self):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.EMA = tf.train.ExponentialMovingAverage(decay=0.99)
            self.cup_models = { i: CupModel(i, self.cup_restore, self.cup_model_path) for i in self.cup_list}

            self.hand_model = HandModel(self.batch_size)
            self.touch_filter = TouchFilter(self.hand_model.n_surf_pts, situation_invariant=self.situation_invariant)
            print('Hand Model #PTS: %d'%self.hand_model.n_surf_pts)        

            self.obs_ewp = {i: self.descriptor(self.obs_z, self.obs_z2, self.cup_models[i], self.obs_penetration_penalty) for i in self.cup_list}
            self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in self.cup_list}
            self.inp_ewp = {i: self.descriptor(self.inp_z, self.inp_z2, self.cup_models[i], self.syn_penetration_penalty) for i in self.cup_list}
            self.syn_zzewpg = {i: self.langevin_dynamics[i](self.inp_z, self.inp_z2) for i in self.cup_list}

            self.descriptor_loss = {i : (tf.reduce_mean(self.obs_ewp[i][0]) + tf.reduce_mean(self.obs_ewp[i][2]) * self.prior_weight) - (tf.reduce_mean(self.inp_ewp[i][0]) + tf.reduce_mean(self.inp_ewp[i][2]) * self.prior_weight) for i in self.cup_list}
    
    def hand_prior_nn(self, hand_z):
        h1 = fully_connected(hand_z, 64, activation_fn=tf.nn.relu, weight_decay=self.l2_reg, scope='hand_prior_1')
        h2 = fully_connected(h1, 64, activation_fn=tf.nn.relu, weight_decay=self.l2_reg, scope='hand_prior_2')
        prior = fully_connected(h2, 1, activation_fn=None, weight_decay=self.l2_reg, scope='hand_prior_3')
        return prior

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
            # gz_abs = tf.reduce_mean(tf.abs(grad_z), axis=0)
            if self.adaptive_langevin:
                # apply_op = self.EMA.apply([gz_abs])
                # with tf.control_dependencies([apply_op]):
                # g_avg = self.EMA.average(gz_abs) + 1e-9
                grad_z = grad_z / self.gz_mean
            if self.clip_norm_langevin:
                grad_z = tf.clip_by_norm(grad_z, 31, axes=-1)
            z2g = 0
            if self.dynamic_z2:
                z2g = tf.gradients(tf.reduce_mean(energy) + tf.reduce_mean(tf.norm(z2, axis=-1)), z2)[0]
                z2g /= tf.norm(z2g, axis=-1, keepdims=True)

            grad_z = grad_z * self.z_weight[0]
            z = z - self.step_size * grad_z * self.update_mask + self.step_size * tf.random.normal(z.shape, mean=0.0, stddev=self.z_weight[0]) * self.update_mask * self.random_strength
            z2 = z2 - self.step_size * z2g + self.step_size * tf.random.normal(z2.shape) * self.random_strength
            return [z, z2, energy, weight, hand_prior]
            
        return langevin_dynamics

    def build_train(self):
        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('model')]
        self.des_optim = tf.train.GradientDescentOptimizer(self.d_lr)
        des_grads_vars = {i : self.des_optim.compute_gradients(self.descriptor_loss[i] + tf.add_n(tf.losses.get_losses()), var_list=self.des_vars) for i in self.cup_list}
        for (g,v) in des_grads_vars[3]:
            if g is None:
                print(v)
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in self.cup_list}
        self.des_train = {i : self.des_optim.apply_gradients(des_grads_vars[i]) for i in self.cup_list}
        self.obs_z2_update = {i : self.obs_z2 - tf.gradients(self.obs_ewp[i][0] + tf.reduce_mean(tf.norm(self.obs_z2, axis=-1)), self.obs_z2)[0] * self.d_lr for i in self.cup_list}
        pass
    
    def build_summary(self):
        self.summ_obs_e = tf.placeholder(tf.float32, [], 'summ_obs_e')
        self.summ_ini_e = tf.placeholder(tf.float32, [], 'summ_ini_e')
        self.summ_syn_e = tf.placeholder(tf.float32, [], 'summ_syn_e')
        self.summ_obs_p = tf.placeholder(tf.float32, [], 'summ_obs_p')
        self.summ_ini_p = tf.placeholder(tf.float32, [], 'summ_ini_p')
        self.summ_syn_p = tf.placeholder(tf.float32, [], 'summ_syn_p')
        self.summ_descriptor_loss = tf.placeholder(tf.float32, [], 'summ_descriptor_loss')
        self.summ_obs_w = tf.placeholder(tf.float32, [None], 'summ_obs_w')
        self.summ_syn_w = tf.placeholder(tf.float32, [None], 'summ_syn_w')
        self.summ_g_avg = tf.placeholder(tf.float32, [31], 'summ_g_avg')
        scalar_summs = [
            tf.summary.scalar('energy/obs', self.summ_obs_e), 
            tf.summary.scalar('energy/ini', self.summ_ini_e), 
            tf.summary.scalar('energy/syn', self.summ_syn_e), 
            tf.summary.scalar('energy/obs', self.summ_obs_e), 
            tf.summary.scalar('prior/obs', self.summ_obs_p), 
            tf.summary.scalar('prior/ini', self.summ_ini_p), 
            tf.summary.scalar('prior/syn', self.summ_syn_p), 
            tf.summary.scalar('prior/imp', self.summ_ini_p - self.summ_syn_p), 
            tf.summary.histogram('weight/obs', self.summ_obs_w),
            tf.summary.histogram('weight/syn', self.summ_syn_w),
            tf.summary.scalar('loss', self.summ_descriptor_loss)
        ]
        scalar_summs += [tf.summary.scalar('g_avg/%d'%i, self.summ_g_avg[i]) for i in range(31)]
        self.summ_obs_bw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_bw')
        self.summ_obs_fw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_fw')
        self.summ_syn_bw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_bw')
        self.summ_syn_fw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_fw')
        self.summ_obs_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_im')
        self.summ_syn_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_im')
        self.summ_syn_e_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_e_im')
        self.summ_syn_p_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_p_im')
        img_summs = [
            tf.summary.image('w/obs/back', self.summ_obs_bw), 
            tf.summary.image('w/obs/front', self.summ_obs_fw), 
            tf.summary.image('w/syn/back', self.summ_syn_bw), 
            tf.summary.image('w/syn/front', self.summ_syn_fw), 
            tf.summary.image('render/obs', self.summ_obs_im), 
            tf.summary.image('render/syn', self.summ_syn_im), 
            tf.summary.image('plot/syn_e', self.summ_syn_e_im), 
            tf.summary.image('plot/syn_prior', self.summ_syn_p_im),
        ]
        self.scalar_summ = tf.summary.merge(scalar_summs)
        self.all_summ = tf.summary.merge(img_summs + scalar_summs)
        pass