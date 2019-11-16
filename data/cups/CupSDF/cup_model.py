import tensorflow as tf

class CupModel:
    def __init__(self, config):
        self.build_config(config)
        self.build_input()
        self.build_model()
        self.build_optim()
        self.build_summary()

    def build_config(self, config):
        self.float = tf.float32
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.delta = config.delta
        self.batch_size = config.batch_size

    def build_input(self):
        self.x = tf.placeholder(self.float, [None, 3])
        self.sdf = tf.placeholder(self.float, [None, 1])
        self.lr = tf.placeholder(self.float, [])

    def clamp(self, x):
        return tf.minimum(self.delta, tf.maximum(x, -self.delta))

    def build_model(self):
        h = self.x
        for i in range(7):
            h = tf.layers.Dense(64, activation='relu', kernel_initializer=tf.initializers.glorot_normal())(h)
            h = tf.layers.Dropout(0.2)(h)
        self.pred = tf.layers.Dense(1)(h)
        self.loss = tf.reduce_sum(tf.abs(self.clamp(self.pred) - self.clamp(self.sdf)))
        self.error = tf.reduce_mean(tf.abs(self.pred - self.sdf))
    
    def build_optim(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, \
            beta1=self.beta1, beta2=self.beta2)
        self.grads_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        self.train_step = optimizer.apply_gradients(self.grads_vars)
    
    def build_summary(self):
        self.grads = tf.concat([tf.reshape(grad, [-1]) for (grad, var) in self.grads_vars], axis=0)
        _ = tf.summary.histogram('grads', self.grads)
        _ = tf.summary.scalar('loss', self.loss)
        _ = tf.summary.scalar('error', self.error)
        self.summ = tf.summary.merge_all()


        # TODO: 1. verify if smaller cup model is equivalent to old model
        #           1.1. if there's a constant scale / shift
        #           1.2. plot to see if dist and grad makes sense
        #       2. verify if smaller cup model is of the same order as cup shapes
        #       3. verify if smaller cup model result is the same as expected on gt examples