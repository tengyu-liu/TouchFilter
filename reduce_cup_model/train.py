import os
import sys

import tensorflow as tf
import numpy as np

from CupModel import CupModel as OldModel
from model import CupModel as NewModel

cup_id = int(sys.argv[1])
total_steps = 100000
batch_size = 102400

new_model = NewModel(cup_id)
old_model = OldModel(cup_id, 999, '/media/tengyu/research/projects/TouchFilter/data/cups/models')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('models/cup_%d'%(cup_id), exist_ok=True)

train_writer = tf.summary.FileWriter('logs/cup_%d'%cup_id, sess.graph)
saver = tf.train.Saver(max_to_keep=0)

for global_step in range(total_steps):
    pts = np.random.normal(size=[batch_size, 3])

    old_pred = sess.run(old_model.pred, feed_dict={old_model.x: pts})
    _, loss, summ = sess.run([new_model.train, new_model.loss, new_model.summ], feed_dict={new_model.x: pts, new_model.gt: old_pred, new_model.d_lr: 1e-3})

    print('\r[%d]: %f, %f'%(global_step, loss, loss / np.mean(np.square(old_pred))), end='')
    if global_step % 10 == 0:
        train_writer.add_summary(summ, global_step=global_step)

saver.save(sess, 'models/cup_%d/%d.ckpt'%(cup_id, global_step))

for global_step in range(total_steps):
    pts = np.random.normal(size=[batch_size, 3])

    old_pred = sess.run(old_model.pred, feed_dict={old_model.x: pts})
    _, loss, summ = sess.run([new_model.train, new_model.loss, new_model.summ], feed_dict={new_model.x: pts, new_model.gt: old_pred, new_model.d_lr: 1e-4})

    print('\r[%d]: %f, %f'%(global_step, loss, loss / np.mean(np.square(old_pred))), end='')
    if global_step % 10 == 0:
        train_writer.add_summary(summ, global_step=global_step + total_steps)

saver.save(sess, 'models/cup_%d/%d.ckpt'%(cup_id, global_step + total_steps))

for global_step in range(total_steps):
    pts = np.random.normal(size=[batch_size, 3])

    old_pred = sess.run(old_model.pred, feed_dict={old_model.x: pts})
    _, loss, summ = sess.run([new_model.train, new_model.loss, new_model.summ], feed_dict={new_model.x: pts, new_model.gt: old_pred, new_model.d_lr: 1e-5})

    print('\r[%d]: %f, %f'%(global_step, loss, loss / np.mean(np.square(old_pred))), end='')
    if global_step % 10 == 0:
        train_writer.add_summary(summ, global_step=global_step + total_steps * 2)

saver.save(sess, 'models/cup_%d/%d.ckpt'%(cup_id, global_step + total_steps * 2))
