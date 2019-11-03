import numpy as np
from TouchFilter import TouchFilter
from sklearn.decomposition import PCA
import tensorflow as tf
import pickle
import os
from HandModel import HandModel
from CupModel import CupModel

class Model:
    def __init__(self, config, stddev):
        self.build_config(config, stddev)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
    
    def build_config(self, config, stddev):
        self.situation_invariant = config.situation_invariant
        self.penalty_strength = config.penalty_strength
        self.adaptive_langevin = config.adaptive_langevin
        self.clip_norm_langevin = config.clip_norm_langevin
        self.hand_z_size = config.z_size
        self.pca_size = config.pca_size
        self.batch_size = config.batch_size
        self.langevin_steps = config.langevin_steps
        self.step_size = config.step_size
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.z_weight = tf.constant(stddev, dtype=tf.float32)

        self.debug = config.debug

        self.cup_num = 10
        if self.debug:
            self.cup_num = 2

        self.use_pca = config.use_pca
        if self.use_pca:
            pca = pickle.load(open(os.path.join(os.path.dirname(__file__), 'pca/pkl%d/pca_%d.pkl'%(self.pca_size, self.hand_z_size)), 'rb'))
            assert isinstance(pca, PCA)
            self.pca_components = tf.constant(pca.components_, dtype=tf.float32)
            self.pca_mean = tf.constant(pca.mean_, dtype=tf.float32)
            self.pca_var = tf.constant(np.sqrt(np.expand_dims(pca.explained_variance_ , axis=-1)), dtype=tf.float32)

        self.cup_model_path = os.path.join(os.path.dirname(__file__), '../data/cups/models')
        self.cup_restore = 999
    
    def build_input(self):

        if self.use_pca:
            self.inp_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + (53 - self.pca_size)], 'input_hand_z')
            self.obs_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + (53 - self.pca_size)], 'obs_z')
        else:
            self.inp_z = tf.placeholder(tf.float32, [self.batch_size, 53], 'input_hand_z')
            self.obs_z = tf.placeholder(tf.float32, [self.batch_size, 53], 'obs_z')

        self.cup_r = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'cup_r')
        self.is_training = tf.placeholder(tf.bool, [], 'is_training')
        pass

    def build_model(self):
        self.EMA = tf.train.ExponentialMovingAverage(decay=0.99999)
        self.cup_models = { 
                            1: CupModel(1, self.cup_restore, self.cup_model_path), 
                            2: CupModel(2, self.cup_restore, self.cup_model_path), 
                            3: CupModel(3, self.cup_restore, self.cup_model_path), 
                            4: CupModel(4, self.cup_restore, self.cup_model_path), 
                            5: CupModel(5, self.cup_restore, self.cup_model_path), 
                            6: CupModel(6, self.cup_restore, self.cup_model_path), 
                            7: CupModel(7, self.cup_restore, self.cup_model_path), 
                            8: CupModel(8, self.cup_restore, self.cup_model_path), 
                            9: CupModel(9, self.cup_restore, self.cup_model_path), 
                            10: CupModel(10, self.cup_restore, self.cup_model_path), 
        }

        self.hand_model = HandModel(self.batch_size)
        self.touch_filter = TouchFilter(self.hand_model.n_surf_pts, situation_invariant=self.situation_invariant, penalty_strength=self.penalty_strength)
        print('Hand Model #PTS: %d'%self.hand_model.n_surf_pts)        
        
        self.obs_ewp = {i: self.descriptor(self.obs_z, self.cup_r, self.cup_models[i], reuse=(i!=1)) for i in range(1,self.cup_num + 1)}
        self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in range(1,self.cup_num + 1)}
        self.inp_ewp = {i: self.descriptor(self.inp_z, self.cup_r, self.cup_models[i], reuse=True) for i in range(1,self.cup_num + 1)}
        self.syn_zewp = {i: self.langevin_dynamics[i](self.inp_z, self.cup_r) for i in range(1,self.cup_num + 1)}

        self.descriptor_loss = {i : (self.obs_ewp[i][0] + self.obs_ewp[i][2]) - (self.inp_ewp[i][0] + self.inp_ewp[i][2]) for i in range(1,self.cup_num + 1)}
        pass
    
    def hand_prior(self, hand_z, reuse=True):
        with tf.variable_scope('des_h', reuse=reuse):
            h = tf.layers.dense(hand_z, 1024, activation=tf.nn.relu)
            h = tf.layers.dense(h, 1024, activation=tf.nn.relu)
            pred = tf.layers.dense(h, 1, activation=tf.nn.sigmoid)
            p = tf.print('Hand Prior:', pred)
            with tf.control_dependencies([p]):
                return pred

    def descriptor(self, hand_z, cup_r, cup_model, reuse=True):
        with tf.variable_scope('des_t', reuse=reuse):
            if self.use_pca:
                z_ = tf.concat([tf.matmul(hand_z[:,:self.hand_z_size], self.pca_var * self.pca_components) + self.pca_mean, hand_z[:,self.hand_z_size:]], axis=1)
            else:
                z_ = hand_z
            jrot = tf.reshape(z_[:,:44], [z_.shape[0], 22, 2])
            grot = tf.reshape(z_[:,44:50], [z_.shape[0], 3, 2])
            gpos = z_[:,50:]
            surf_pts = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
            surf_pts = tf.concat(list(surf_pts.values()), axis=1)

            # touch response
            if self.situation_invariant:
                energy, weight = self.touch_filter(surf_pts, cup_model, cup_r)
            else:
                energy, weight = self.touch_filter(surf_pts, cup_model, cup_r, hand_z)
            
            hand_prior = tf.reduce_mean(self.hand_prior(hand_z, reuse=reuse))
            return energy, weight, hand_prior

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, r):
            energy, weight, hand_prior = self.descriptor(z,r,self.cup_models[cup_id],reuse=True) #+ tf.reduce_mean(z[:,:self.hand_z_size] * z[:,:self.hand_z_size]) + tf.reduce_mean(z[:,self.hand_z_size:] * z[:,self.hand_z_size:])
            grad_z = tf.gradients(energy + hand_prior, z)[0]
            gz_abs = tf.reduce_mean(tf.abs(grad_z), axis=0)
            if self.adaptive_langevin:
                apply_op = self.EMA.apply([gz_abs])
                with tf.control_dependencies([apply_op]):
                    g_avg = self.EMA.average(gz_abs) + 1e-9
                    grad_z = grad_z / g_avg
            if self.clip_norm_langevin:
                grad_z = tf.clip_by_norm(grad_z, 1, axes=-1)

            grad_z = grad_z * self.z_weight
            # p = tf.print('GRAD: ', grad_z, summarize=-1)
            # with tf.control_dependencies([p]):
            z = z - self.step_size * grad_z # + self.step_size * tf.random.normal(z.shape, mean=0.0, stddev=self.z_weight)
            return [z, energy, weight, hand_prior]
            
        return langevin_dynamics

    def build_train(self):
        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_optim = tf.train.GradientDescentOptimizer(self.d_lr)
        des_grads_vars = {i : self.des_optim.compute_gradients(self.descriptor_loss[i], var_list=self.des_vars) for i in range(1,self.cup_num + 1)}
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in range(1,self.cup_num + 1)}
        self.des_train = {i : self.des_optim.apply_gradients(des_grads_vars[i]) for i in range(1,self.cup_num + 1)}
        pass
    
    def build_summary(self):
        self.summ_obs_e = tf.placeholder(tf.float32, [], 'summ_obs_e')
        self.summ_ini_e = tf.placeholder(tf.float32, [], 'summ_ini_e')
        self.summ_syn_e = tf.placeholder(tf.float32, [], 'summ_syn_e')
        self.summ_obs_p = tf.placeholder(tf.float32, [], 'summ_obs_p')
        self.summ_ini_p = tf.placeholder(tf.float32, [], 'summ_ini_p')
        self.summ_syn_p = tf.placeholder(tf.float32, [], 'summ_syn_p')
        self.summ_descriptor_loss = tf.placeholder(tf.float32, [], 'summ_descriptor_loss')
        _ = tf.summary.scalar('energy/obs', self.summ_obs_e)
        _ = tf.summary.scalar('energy/ini', self.summ_ini_e)
        _ = tf.summary.scalar('energy/syn', self.summ_syn_e)
        _ = tf.summary.scalar('energy/obs', self.summ_obs_e)
        _ = tf.summary.scalar('prior/ini', self.summ_ini_p)
        _ = tf.summary.scalar('prior/syn', self.summ_syn_p)
        _ = tf.summary.scalar('prior/imp', self.summ_ini_p - self.summ_syn_p)
        _ = tf.summary.scalar('loss', self.summ_descriptor_loss)
        self.summ_obs_bw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_bw')
        self.summ_obs_fw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_fw')
        self.summ_syn_bw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_bw')
        self.summ_syn_fw = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_fw')
        _ = tf.summary.image('summ_obs_back_w', self.summ_obs_bw)
        _ = tf.summary.image('summ_obs_front_w', self.summ_obs_fw)
        _ = tf.summary.image('summ_syn_back_w', self.summ_syn_bw)
        _ = tf.summary.image('summ_syn_front_w', self.summ_syn_fw)
        self.summ_obs_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_obs_im')
        self.summ_syn_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_im')
        self.summ_syn_e_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_e_im')
        self.summ_syn_p_im = tf.placeholder(tf.uint8, [None, 480, 640, 3], 'summ_syn_p_im')
        _ = tf.summary.image('summ_obs_im', self.summ_obs_im)
        _ = tf.summary.image('summ_syn_im', self.summ_syn_im)
        _ = tf.summary.image('summ_syn_e_im', self.summ_syn_e_im)
        _ = tf.summary.image('summ_syn_p_im', self.summ_syn_p_im)
        self.summ = tf.summary.merge_all()
        pass
