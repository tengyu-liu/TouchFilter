import tensorflow as tf

class CupModel:
    def __init__(self, cup_id):
        with tf.variable_scope('cup_%d'%cup_id):
            self.x = tf.placeholder(tf.float32, [None, 3], 'input')
            h1 = tf.layers.dense(self.x, 128, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu)
            h3 = tf.layers.dense(h2, 128, activation=tf.nn.relu)
            self.out = tf.layers.dense(h3, 1)
            self.gt = tf.placeholder(tf.float32, [None, 1], 'gt')
            self.loss = tf.reduce_mean(tf.math.square(self.out - self.gt))

            self.d_lr = tf.placeholder(tf.float32, [], 'lr')
            self.beta1 = 0.99
            self.beta2 = 0.999

            self.des_vars = [var for var in tf.trainable_variables()]
            self.des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
            des_grads_vars = self.des_optim.compute_gradients(self.loss, var_list=self.des_vars)
            self.train = self.des_optim.apply_gradients(des_grads_vars)

            _ = tf.summary.scalar('loss', self.loss)
            _ = tf.summary.scalar('lr', self.d_lr)
            self.summ = tf.summary.merge_all()