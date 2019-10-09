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
        self.build_summary()
    
    def build_config(self, config):
        self.n_pts = 20000
        self.situation_invariant = config.situation_invariant
        self.penalty_strength = config.penalty_strength
        self.adaptive_langevin = config.adaptive_langevin
        self.clip_norm_langevin = config.clip_norm_langevin
        self.hand_z_size = config.z_size
        self.pca_size = config.pca_size
        self.batch_size = config.batch_size
        
        self.use_pca = config.use_pca
        if self.use_pca:
            pca = pickle.load(open(os.path.join(os.path.dirname(__file__), 'pca/pkl%d/pca_%d.pkl'%(self.pca_size, self.hand_z_size)), 'rb'))
            assert isinstance(pca, PCA)
            self.pca_var = pca.explained_variance_
            self.pca_mean = pca.mean_
            self.pca_components = pca.components_

        self.cup_model_path = os.path.join(os.path.dirname(__file__), '../data/cups/models')
        self.cup_restore = 999
    
    def build_input(self):
        self.ini_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size], 'input_hand_z')
        self.cup_r = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'cup_r')
        self.obs_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size], 'obs_z')

        pass

    def build_model(self):
        self.EMA = tf.train.ExponentialMovingAverage(decay=0.9999)
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

        self.obs_e = {i: self.descriptor(self.obs_z, self.cup_r, self.cup_models[i], reuse=(i!=1)) for i in range(1,11)}
        self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in range(1,11)}
        self.ini_e = {i: self.descriptor(self.ini_z, self.cup_r, self.cup_models[i], reuse=True) for i in range(1,11)}
        self.syn_z = {i: self.langevin_dynamics[i](self.ini_z, self.cup_r) for i in range(1,11)}
        self.syn_e = {i: self.descriptor(self.syn_z, self.cup_r, self.cup_models[i], reuse=True) for i in range(1,11)}

        self.descriptor_loss = {i : self.syn_e[i] - self.obs_e[i] for i in range(1,11)}

        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
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
            energy = self.touch_filter(pts, m, r)
            return energy

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, r):
            def _cond(z,r,i):
                return tf.less(i, self.langevin_steps)

            def _body(z,r,i):
                energy = self.descriptor(z,r,self.cup_models[cup_id],reuse=True) + z * z
                grad_z = tf.gradients(energy, z)[0]
                if self.adaptive_langevin:
                    grad_z = grad_z / self.mean_gradient
                if self.clip_norm_langevin:
                    grad_z = tf.clip_by_norm(grad_z, 1)
                z = z + self.step_size * grad_z # + tf.random.normal(z.shape, mean=0.0, stddev=1e-3)
                return z, r, tf.add(i, 1)
            
            with tf.name_scope('langevin_dynamics'):
                i = tf.constant(0)
                z, r, i = tf.while_loop(_cond, _body, [z, r, i])
                return z
        return langevin_dynamics

    def build_train(self):
        self.des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        des_grads_vars = {i : self.des_optim.compute_gradients(self.descriptor_loss[i], var_list=self.des_vars) for i in range(1,11)}
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in range(1,11)}
        self.des_train = {i : self.des_optim.apply_gradients(des_grads_vars[i]) for i in range(1,11)}
        pass
    
    def build_summary(self):
        self.summ_obs_e = tf.placeholder(tf.float32, [], 'summ_obs_e')
        self.summ_ini_e = tf.placeholder(tf.float32, [], 'summ_ini_e')
        self.summ_syn_e = tf.placeholder(tf.float32, [], 'summ_syn_e')
        self.summ_descriptor_loss = tf.placeholder(tf.float32, [], 'summ_descriptor_loss')
        _ = tf.summary.scalar('energy/obs', self.summ_obs_e)
        _ = tf.summary.scalar('energy/ini', self.summ_ini_e)
        _ = tf.summary.scalar('energy/syn', self.summ_syn_e)
        _ = tf.summary.scalar('energy/imp', self.summ_descriptor_loss)
        self.summ = tf.summary.merge_all()
        pass
