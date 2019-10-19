import os
import sys

import tensorflow as tf
import numpy as np

from CupModel import CupModel as OldModel
from model import CupModel as NewModel

cup_id = int(sys.argv[1])

total_steps = 10000
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

plateu = 20

lr = 1e-3
global_step = 0

global_minimum = float('inf')
global_minimum_step = -1

while lr > 1e-9:
	pts = np.random.normal(size=[batch_size, 3])

	old_pred = sess.run(old_model.pred, feed_dict={old_model.x: pts})
	_, loss, summ = sess.run([new_model.train, new_model.loss, new_model.summ], feed_dict={new_model.x: pts, new_model.gt: old_pred, new_model.d_lr: 1e-3})

	print('\r[%d]: LR: %f, Err: %f, Rel. Err.: %f'%(global_step, lr, loss, loss / np.mean(np.square(old_pred))), end='')

	if loss < global_minimum: 
		global_minimum = loss
		global_minimum_step = global_step
	else:
		if global_step > global_minimum_step + plateu:
			lr *= 0.1
			global_minimum_step = global_step

	global_step += 1
	
saver.save(sess, 'models/cup_%d.ckpt'%(cup_id, global_step + total_steps * 3))
