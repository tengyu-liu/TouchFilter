import tensorflow as tf
import numpy as np
import os
import pickle
import random

import trimesh as tm

from cup_model import CupModel
from config import config

cup = tm.load(os.path.join(os.path.dirname(__file__), '../onepiece/%d.obj'%config.cup_id))

model = CupModel(config)

c = tf.ConfigProto()
c.gpu_options.allow_growth = True

sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('logs/%s-[cup%d]'%(config.name, config.cup_id))

batch_size = config.batch_size
saver = tf.train.Saver()

step = 0

for epoch in range(config.epochs):

    losses = []

    for batch_id in range(2000):
        batch_x = np.random.random([config.batch_size, 3]) * 0.6 - 0.3
        batch_y = tm.proximity.signed_distance(cup, batch_x)

        loss, error, _, w_grad, summ = sess.run([
            model.loss, 
            model.error,
            model.train_step, 
            model.grads,
            model.summ
            ], feed_dict={
            model.x: batch_x,
            model.sdf: batch_y,
            model.lr: config.d_lr
        })

        losses.append(loss)
        print('\r%d: %d/%d Loss: %f, Error: %f, WGrad: %f'%(epoch, batch_id, batch_num, loss, error, np.mean(np.square(w_grad))), end='')
        train_writer.add_summary(summ, global_step=step)
        step += 1

    print('\n[%d] Loss: %f'%(epoch, np.mean(losses)))
    saver.save(sess, 'models/%s[cup_%d]_%d.ckpt'%(config.name, config.cup_id, epoch))
