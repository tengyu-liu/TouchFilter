import pickle
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
from HandModel import HandModel
from CupModel import CupModel
from TouchFilter import TouchFilter

class Model:
    def __init__(self, config):
        self.config = config
        self.build_config()
        self.build_input()
        self.build_model()
        self.build_train()
        self.build_summary()
        pass
    
    def build_config(self):
        self.use_generator = self.config.use_generator
        self.batch_size = self.config.batch_size
        self.langevin_steps = self.config.langevin_steps
        self.step_size = self.config.step_size
        self.n_filter = self.config.n_filter
        self.n_channel = self.config.n_channel
        self.use_pca = self.config.use_pca
        self.des_weight_norm_mult = self.config.des_weight_norm
        self.gen_weight_norm_mult = self.config.gen_weight_norm

        self.d_lr = self.config.d_lr
        self.g_lr = self.config.g_lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2

        self.vae_z_size = 10

        self.cup_model_path = 'data/cups/models'
        self.cup_restore = 999

        if self.use_pca:
            self.hand_z_size = self.config.z_size
            pca = pickle.load(open('hand_prior/pca_%d.pkl'%self.hand_z_size, 'rb'))
            self.pca_components = tf.constant(pca.components_, dtype=tf.float32)
            self.pca_mean = tf.constant(pca.mean_, dtype=tf.float32)
            self.pca_var = tf.constant(np.sqrt(np.expand_dims(pca.explained_variance_ , axis=-1)), dtype=tf.float32)
        else:
            self.hand_z_size = 53
        
        self.N_PTS = 158

        axis = np.linspace(-0.3, 0.3, 64)
        pts = np.transpose(np.stack(np.meshgrid(axis, axis, axis), axis=-1), [1,0,2,3])
        self.pts = tf.constant(pts.reshape([-1,3]), dtype=tf.float32)

        pass

    def build_input(self):
        self.cup_r = tf.placeholder(tf.float32, [self.batch_size, 3, 3], 'cup_r')
        self.obs_z = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + 9], 'gt_z')
        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.hand_z_size + 9], 'z_input')

        self.obs_touch_energy_summ_in = tf.placeholder(tf.float32, [], 'obs_touch_energy_summ_in')
        self.obs_prior_energy_summ_in = tf.placeholder(tf.float32, [], 'obs_prior_energy_summ_in')
        self.syn_touch_energy_summ_in = tf.placeholder(tf.float32, [], 'syn_touch_energy_summ_in')
        self.syn_prior_energy_summ_in = tf.placeholder(tf.float32, [], 'syn_prior_energy_summ_in')
        self.improved_touch_energy_summ_in = tf.placeholder(tf.float32, [], 'improved_touch_energy_summ_in')
        self.improved_prior_energy_summ_in = tf.placeholder(tf.float32, [], 'improved_prior_energy_summ_in')
        self.descriptor_loss_summ_in = tf.placeholder(tf.float32, [], 'descriptor_loss_summ_in')
        self.generator_loss_summ_in = tf.placeholder(tf.float32, [], 'generator_loss_summ_in')
        self.des_weight_norm_summ_in = tf.placeholder(tf.float32, [], 'des_weight_norm_summ_in')
        self.gen_weight_norm_summ_in = tf.placeholder(tf.float32, [], 'gen_weight_norm_summ_in')

        self.random_in = tf.placeholder(tf.float32, [self.batch_size, self.vae_z_size], 'random_in')
        pass

    def build_model(self):
        self.touch_filter = TouchFilter(self.N_PTS, n_filters=self.n_filter, prange=0.1, pnum=20, pexp=2, nrange=-0.1, nnum=20, nexp=2, n_channel=self.n_channel)
        self.hand_model = HandModel(self.batch_size)
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

        _ = self.descriptor(self.obs_z, self.cup_r, self.cup_models[1], reuse=False)
        _ = self.generator(self.cup_r, self.cup_models[1], reuse=False)

        self.langevin_dynamics = {i : self.langevin_dynamics_fn(i) for i in range(1,11)}

        self.obs_touch_energy = { i : tf.reduce_mean(self.touch_descriptor(self.obs_z, self.cup_r, self.cup_models[i], reuse=True)) for i in range(1,11) }
        self.obs_prior_energy = { i : tf.reduce_mean(self.prior_descriptor(self.obs_z, reuse=True)) for i in range(1,11) }

        # 0. Generate Z with G
        self.initial_z = { i : self.generator(self.cup_r, self.cup_models[i], reuse=True) for i in range(1,11) }
        self.initial_touch_energy = { i : tf.reduce_mean(self.touch_descriptor(self.initial_z[i], self.cup_r, self.cup_models[i], reuse=True)) for i in range(1,11) }
        self.initial_prior_energy = { i : tf.reduce_mean(self.prior_descriptor(self.initial_z[i], reuse=True)) for i in range(1,11) }

        # 1. Synthesize Z' with D
        self.syn_z = { i : self.langevin_dynamics[i](self.z_input, self.cup_r) for i in range(1,11) }
        self.syn_touch_energy = { i : tf.reduce_mean(self.touch_descriptor(self.z_input, self.cup_r, self.cup_models[i], reuse=True)) for i in range(1,11) }
        self.syn_prior_energy = { i : tf.reduce_mean(self.prior_descriptor(self.z_input, reuse=True)) for i in range(1,11) }

        # 2. Train D with Z'-Z
        self.descriptor_loss = {i : self.syn_touch_energy[i] + self.syn_prior_energy[i] - self.obs_touch_energy[i] - self.obs_prior_energy[i] for i in range(1,11)}

        # 3. Train G with -Z'
        if self.use_generator:
            self.generator_loss = {i : tf.reduce_mean(tf.square(self.z_input - self.initial_z[i])) for i in range(1,11)}

        if self.use_pca:
            z_ = tf.concat([tf.matmul(self.obs_z[:,:self.hand_z_size], self.pca_var * self.pca_components) + self.pca_mean, self.obs_z[:,self.hand_z_size:]], axis=-1)
        else:
            z_ = self.obs_z

        jrot = tf.reshape(z_[:,:44], [z_.shape[0], 22, 2])
        grot = tf.reshape(z_[:,44:50], [z_.shape[0], 3, 2])
        gpos = z_[:,50:]
        self.obs_pts, _, _ = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)

        # # debug
        # cup_res = {i : self.touch_filter.debug(self.obs_pts, _, self.cup_models[i], self.cup_r) for i in range(1,11)}
        # self.cup_pts = {i : cup_res[i][0] for i in cup_res}
        # self.cup_val = {i : cup_res[i][1] for i in cup_res}

        pass
    
    def build_train(self):
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]
        self.des_weight_norm = sum(tf.nn.l2_loss(v) for v in des_vars)
        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        des_grads_vars = {i : des_optim.compute_gradients(self.descriptor_loss[i] + self.des_weight_norm * self.des_weight_norm_mult, var_list=des_vars) for i in range(1,11)}
        des_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in des_grads_vars[i]] for i in range(1,11)}
        self.des_train = {i : des_optim.apply_gradients(des_grads_vars[i]) for i in range(1,11)}
        
        if self.use_generator:
            gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]
            self.gen_weight_norm = sum(tf.nn.l2_loss(v) for v in gen_vars)
            gen_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
            gen_grads_vars = {i : gen_optim.compute_gradients(self.generator_loss[i] + self.gen_weight_norm * self.gen_weight_norm_mult, var_list=gen_vars) for i in range(1,11)}
            gen_grads_vars = {i : [(tf.clip_by_norm(g,1), v) for (g,v) in gen_grads_vars[i]] for i in range(1,11)}
            self.gen_train = {i : gen_optim.apply_gradients(gen_grads_vars[i]) for i in range(1,11)}
        pass

    def build_summary(self):
        tf.summary.scalar('obs_touch_energy', self.obs_touch_energy_summ_in)
        tf.summary.scalar('obs_prior_energy', self.obs_prior_energy_summ_in)
        tf.summary.scalar('syn_touch_energy', self.syn_touch_energy_summ_in)
        tf.summary.scalar('syn_prior_energy', self.syn_prior_energy_summ_in)
        tf.summary.scalar('improved_touch_energy', self.improved_touch_energy_summ_in)
        tf.summary.scalar('improved_prior_energy', self.improved_prior_energy_summ_in)
        tf.summary.scalar('descriptor_loss', self.descriptor_loss_summ_in)
        tf.summary.scalar('des_weight_norm', self.des_weight_norm_summ_in)
        if self.use_generator:
            tf.summary.scalar('gen_weight_norm', self.gen_weight_norm_summ_in)
            tf.summary.scalar('generator_loss', self.generator_loss_summ_in)
        self.summaries = tf.summary.merge_all()
        pass

    def generator(self, r, m, reuse=False):
        with tf.variable_scope('gen', reuse=reuse):
            if self.use_generator:
                cup_pts = tf.transpose(tf.matmul(tf.transpose(r, perm=[0,2,1]), tf.transpose(tf.tile(tf.expand_dims(self.pts, axis=0), [self.batch_size, 1, 1]), perm=[0,2,1])), perm=[0,2,1]) * 5
                cup_val = m.predict(tf.reshape(cup_pts, [self.batch_size * 64 * 64 * 64, 3]))
                
                cup_d = tf.reshape(cup_val[:,0], [self.batch_size, 64*64*64, 1])
                cup_g = tf.reshape(cup_val[:,1:], [self.batch_size, 64*64*64,3])
                cup_g = tf.transpose(tf.matmul(r, tf.transpose(cup_g, perm=[0,2,1])), perm=[0,2,1])

                cup_val = tf.reshape(tf.concat([cup_d, cup_g], axis=-1), [self.batch_size, 64, 64, 64, 4])

                cup_val = tf.layers.MaxPooling3D(4, 4)(cup_val)
                conv1 = tf.layers.Conv3D(64, [6,6,6])(cup_val)
                conv1 = tf.nn.leaky_relu(conv1)
                conv2 = tf.layers.Conv3D(64, [8,8,8])(conv1)
                conv2 = tf.nn.leaky_relu(conv2)
                conv3 = tf.layers.Conv3D(64, [4,4,4])(conv2)
                h = tf.reshape(conv3, [self.batch_size, 64])
                h = tf.nn.leaky_relu(h)
                # h = tf.reshape(h, [self.batch_size, 64])
                # h = tf.nn.leaky_relu(h)
                h = tf.layers.Dense(self.vae_z_size * 2)(h)
                h = tf.reshape(h, [self.batch_size, self.vae_z_size, 2])
                mu = h[:,:,0]
                sigma = h[:,:,1]
                h = self.random_in * sigma * sigma + mu
                h = tf.reshape(h, [self.batch_size, self.vae_z_size])
                h = tf.nn.leaky_relu(h)
                h = tf.layers.Dense(self.hand_z_size + 9)(h)
                return h
            else:
                return tf.random.normal([self.batch_size, self.hand_z_size + 9])
        
    def descriptor(self, z, r, m, reuse=False):
        return self.touch_descriptor(z,r,m,reuse) + self.prior_descriptor(z, reuse)

    def touch_descriptor(self, z, r, m, reuse=False):
        with tf.variable_scope('des_t', reuse=reuse):
            if self.use_pca:
                z_ = tf.concat([tf.matmul(z[:,:self.hand_z_size], self.pca_var * self.pca_components) + self.pca_mean, z[:,self.hand_z_size:]], axis=1)
            else:
                z_ = z
            jrot = tf.reshape(z_[:,:44], [z_.shape[0], 22, 2])
            grot = tf.reshape(z_[:,44:50], [z_.shape[0], 3, 2])
            gpos = z_[:,50:]
            pts, vecs, surf_pts = self.hand_model.tf_forward_kinematics(gpos, grot, jrot)
            surf_pts = tf.concat(list(surf_pts.values()), axis=1)

            # touch response
            touch_response = self.touch_filter(pts, vecs, m, r)
            touch_response = tf.reshape(touch_response, [self.batch_size, self.touch_filter.n_pts * self.touch_filter.n_filters])
            
            # penetration response
            surf_pts = tf.transpose(tf.matmul(
                tf.transpose(r, perm=[0,2,1]), 
                tf.transpose(surf_pts, perm=[0,2,1])), perm=[0, 2, 1]) * 4
            penetration_response = tf.reshape(m.predict(tf.reshape(surf_pts, [-1,3])), [surf_pts.shape[0], self.hand_model.n_surf_pts, 4])[..., 0]
            
            h = tf.concat([touch_response, penetration_response], -1)
            h = tf.layers.dense(h, units=128)
            h = tf.nn.leaky_relu(h)
            h = tf.layers.dense(h, units=128)
            h = tf.nn.leaky_relu(h)
            energy = tf.layers.dense(h, units=1)
            return energy
        
    def prior_descriptor(self, z, reuse=False):
        with tf.variable_scope('des_p', reuse=reuse):
            return tf.reduce_sum(tf.math.square(z))

    def langevin_dynamics_fn(self, cup_id):
        def langevin_dynamics(z, r):
            def _cond(z,r,i):
                return tf.less(i, self.langevin_steps)

            def _body(z,r,i):
                energy = self.descriptor(z,r,self.cup_models[cup_id],reuse=True)
                grad_z = tf.gradients(energy, z)[0]
                grad_z = tf.clip_by_norm(grad_z, 1)
                z = z + self.step_size * grad_z + tf.random.normal(z.shape, mean=0.0, stddev=1e-3)
                return z, r, tf.add(i, 1)
            
            with tf.name_scope('langevin_dynamics'):
                i = tf.constant(0)
                z, r, i = tf.while_loop(_cond, _body, [z, r, i])
                return z
        return langevin_dynamics