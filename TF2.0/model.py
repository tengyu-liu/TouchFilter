import numpy as np
from TouchFilter import TouchFilter
from sklearn.decomposition import PCA
import tensorflow as tf
import pickle
import os
from HandModel import HandModel
from CupModel import CupModel

class Model:
    def __init__(self, config):
        self.build_config(config)
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
    
    def build_config(self, config):
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
        self.inp_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + (53 - self.pca_size)], 'input_hand_z')
        self.cup_r = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'cup_r')
        self.obs_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + (53 - self.pca_size)], 'obs_z')
        self.stddev = tf.placeholder(tf.float32, [], 'stddev')
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
        
        self.obs_ew = {i: self.descriptor(self.obs_z, self.cup_r, self.cup_models[i], reuse=(i!=1)) for i in range(1,self.cup_num + 1)}
        self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in range(1,self.cup_num + 1)}
        self.inp_ew = {i: self.descriptor(self.inp_z, self.cup_r, self.cup_models[i], reuse=True) for i in range(1,self.cup_num + 1)}
        self.syn_zew = {i: self.langevin_dynamics[i](self.inp_z, self.cup_r) for i in range(1,self.cup_num + 1)}

        self.descriptor_loss = {i : self.obs_ew[i][0] - self.inp_ew[i][0] for i in range(1,self.cup_num + 1)}
        pass
    
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
            return energy, weight

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, r):
            z_weight = tf.constant([10, 10, 2, 2, 2, 2, 2, 2, 1, 1, 1], dtype=tf.float32)
            energy, weight = self.descriptor(z,r,self.cup_models[cup_id],reuse=True) #+ tf.reduce_mean(z[:,:self.hand_z_size] * z[:,:self.hand_z_size]) + tf.reduce_mean(z[:,self.hand_z_size:] * z[:,self.hand_z_size:])
            energy = energy + tf.reduce_mean(tf.norm(z / z_weight, axis=-1)) * 3
            grad_z = tf.gradients(energy, z)[0]
            gz_abs = tf.abs(grad_z)
            if self.adaptive_langevin:
                apply_op = self.EMA.apply([gz_abs])
                with tf.control_dependencies([apply_op]):
                    g_avg = self.EMA.average(gz_abs) + 1e-6
                    grad_z = grad_z / g_avg
            if self.clip_norm_langevin:
                grad_z = tf.clip_by_norm(grad_z, 1)

            grad_z = grad_z * z_weight
            # p = tf.print('GRAD: ', grad_z, summarize=-1)
            with tf.control_dependencies([p]):
                z = z - self.step_size * grad_z # + tf.random.normal(z.shape, mean=0.0, stddev=self.stddev)
            return [z, energy, weight]
            
        return langevin_dynamics

    def build_train(self):
        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        des_grads_vars = {i : self.des_optim.compute_gradients(self.descriptor_loss[i], var_list=self.des_vars) for i in range(1,self.cup_num + 1)}
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in range(1,self.cup_num + 1)}
        self.des_train = {i : self.des_optim.apply_gradients(des_grads_vars[i]) for i in range(1,self.cup_num + 1)}
        pass
    
    def build_summary(self):
        self.summ_obs_e = tf.placeholder(tf.float32, [], 'summ_obs_e')
        self.summ_ini_e = tf.placeholder(tf.float32, [], 'summ_ini_e')
        self.summ_syn_e = tf.placeholder(tf.float32, [], 'summ_syn_e')
        self.summ_descriptor_loss = tf.placeholder(tf.float32, [], 'summ_descriptor_loss')
        _ = tf.summary.scalar('energy/obs', self.summ_obs_e)
        _ = tf.summary.scalar('energy/ini', self.summ_ini_e)
        _ = tf.summary.scalar('energy/syn', self.summ_syn_e)
        _ = tf.summary.scalar('energy/imp', self.summ_ini_e - self.summ_syn_e)
        _ = tf.summary.scalar('loss', self.summ_descriptor_loss)
        self.summ = tf.summary.merge_all()
        pass
