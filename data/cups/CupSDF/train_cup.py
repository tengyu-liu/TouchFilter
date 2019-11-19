import tensorflow as tf
import numpy as np
import os
import pickle
import random

import trimesh as tm

from cup_model import CupModel
from config import config

model = CupModel(config)

data = np.load('cup_%d.npy'%config.cup_id)
np.random.shuffle(data)
train_data = data[:500000]
test_data = data[500000:]

c = tf.ConfigProto()
c.gpu_options.allow_growth = True

sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter('logs/%s-[cup%d]'%(config.name, config.cup_id))

batch_size = config.batch_size
train_batch_num = int(len(train_data) / batch_size)
test_batch_num = int(len(test_data) / batch_size)

saver = tf.train.Saver()

step = 0

lr = config.d_lr

for epoch in range(config.epochs):

    losses = []
    errors = []
    idx = np.random.permutation(len(train_data))

    for batch_id in range(train_batch_num):
        batch = train_data[idx[batch_id * batch_size : (batch_id + 1) * batch_size], :]

        loss, error, _, w_grad, summ = sess.run([
            model.loss, 
            model.error,
            model.train_step, 
            model.grads,
            model.summ
            ], feed_dict={
            model.x: batch[:,:3],
            model.sdf: batch[:,[3]],
            model.lr: lr
        })

        losses.append(loss)
        errors.append(error)
        print('\r%d: %d/%d Loss: %f, Error: %f, WGrad: %f'%(epoch, batch_id, train_batch_num, loss, error, np.mean(np.square(w_grad))), end='', flush=True)
        train_writer.add_summary(summ, global_step=step)
        step += 1

    eval_losses = []
    eval_errors = []
    idx = np.random.permutation(len(test_data))
    for batch_id in range(test_batch_num):
        batch = test_data[idx[batch_id * batch_size : (batch_id + 1) * batch_size], :]
        loss, error = sess.run([
            model.loss, 
            model.error
            ], feed_dict={
            model.x: batch[:,:3],
            model.sdf: batch[:,[3]]
        })
        eval_losses.append(loss)
        eval_errors.append(error)
    print('\n[%d] Loss: %f, Error: %f, Eval Loss: %f, Eval Error: %f'%(epoch, np.mean(losses), np.mean(errors), np.mean(eval_losses), np.mean(eval_errors)))
    saver.save(sess, 'models/%s[cup_%d]_%d.ckpt'%(config.name, config.cup_id, epoch))
    