import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from utils.CupModel import CupModel
from utils.HandModel import HandModel
from utils.TouchFilter import TouchFilter


class Model:
    def __init__(self, config, mean, stddev, cup_list):
        self.build_config(config, mean, stddev, cup_list)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
    
    def build_config(self, config, mean, stddev, cup_list):
        self.situation_invariant = config.situation_invariant
        self.adaptive_langevin = config.adaptive_langevin
        self.clip_norm_langevin = config.clip_norm_langevin
        self.batch_size = config.batch_size
        self.langevin_steps = config.langevin_steps
        self.step_size = config.step_size
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.z_mean = tf.constant(mean, dtype=tf.float32)
        self.z_weight = tf.constant(stddev, dtype=tf.float32)

        self.debug = config.debug

        self.cup_list = cup_list

        self.cup_model_path = os.path.join(os.path.dirname(__file__), '../data/cups/models')
        self.cup_restore = 199
    
    def build_input(self):

        self.inp_z = tf.placeholder(tf.float32, [self.batch_size, 31], 'input_hand_z')
        self.obs_z = tf.placeholder(tf.float32, [self.batch_size, 31], 'obs_z')

        self.cup_r = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'cup_r')
        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.update_mask = tf.placeholder(tf.float32, [self.batch_size, 31], 'update_mask')
        pass

    def build_model(self):
        self.EMA = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.cup_models = { i: CupModel(i, self.cup_restore, self.cup_model_path) for i in self.cup_list}

        self.hand_model = HandModel(self.batch_size)
        self.touch_filter = TouchFilter(self.hand_model.n_surf_pts, situation_invariant=self.situation_invariant)
        print('Hand Model #PTS: %d'%self.hand_model.n_surf_pts)        
        
        self.obs_ewp = {i: self.descriptor(self.obs_z, self.cup_r, self.cup_models[i], reuse=(i!=1)) for i in self.cup_list}
        self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in self.cup_list}
        self.inp_ewp = {i: self.descriptor(self.inp_z, self.cup_r, self.cup_models[i], reuse=True) for i in self.cup_list}
        self.syn_zewp = {i: self.langevin_dynamics[i](self.inp_z, self.cup_r) for i in self.cup_list}

        self.descriptor_loss = {i : self.obs_ewp[i][0] - self.inp_ewp[i][0] for i in self.cup_list}
        pass
    
    def hand_prior(self, hand_z, reuse=True):
        with tf.variable_scope('des_h', reuse=reuse):
            return tf.norm((hand_z - self.z_mean / self.z_weight)[:,-3:], axis=-1)

    def descriptor(self, hand_z, cup_r, cup_model, reuse=True):
        with tf.variable_scope('des_t', reuse=reuse):
            z_ = hand_z
            jrot = z_[:,:22]
            grot = tf.reshape(z_[:,22:28], [z_.shape[0], 3, 2])
            gpos = z_[:,28:]
            surf_pts, surf_normals = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
            surf_pts = tf.concat(list(surf_pts.values()), axis=1)
            surf_normals = tf.concat(list(surf_normals.values()), axis=1)

            # touch response
            energy, weight = self.touch_filter(surf_pts, surf_normals, self.hand_model.pts_feature, cup_model, cup_r)
            
            hand_prior = self.hand_prior(hand_z, reuse=reuse)
            return energy, weight, tf.reduce_mean(hand_prior)

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, r):
            energy, weight, hand_prior = self.descriptor(z,r,self.cup_models[cup_id],reuse=True) #+ tf.reduce_mean(z[:,:self.hand_z_size] * z[:,:self.hand_z_size]) + tf.reduce_mean(z[:,self.hand_z_size:] * z[:,self.hand_z_size:])
            grad_z = tf.gradients(energy, z)[0]
            gz_abs = tf.reduce_mean(tf.abs(grad_z), axis=0)
            if self.adaptive_langevin:
                apply_op = self.EMA.apply([gz_abs])
                with tf.control_dependencies([apply_op]):
                    g_avg = self.EMA.average(gz_abs) + 1e-9
                    grad_z = grad_z / g_avg
            if self.clip_norm_langevin:
                grad_z = tf.clip_by_norm(grad_z, 31, axes=-1)

            grad_z = grad_z * self.z_weight[0]
            # p = tf.print('GRAD: ', grad_z, summarize=-1)
            # with tf.control_dependencies([p]):
            z = z - self.step_size * grad_z * self.update_mask # + self.step_size * tf.random.normal(z.shape, mean=0.0, stddev=self.z_weight[0]) * self.update_mask
            return [z, energy, weight, hand_prior]
            
        return langevin_dynamics

    def build_train(self):
        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_optim = tf.train.GradientDescentOptimizer(self.d_lr)
        des_grads_vars = {i : self.des_optim.compute_gradients(self.descriptor_loss[i], var_list=self.des_vars) for i in self.cup_list}
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in self.cup_list}
        self.des_train = {i : self.des_optim.apply_gradients(des_grads_vars[i]) for i in self.cup_list}
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
            tf.summary.scalar('loss', self.summ_descriptor_loss),
        ]
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
