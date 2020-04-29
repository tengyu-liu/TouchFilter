import os
import pickle
import shutil
import sys

import numpy as np
import tensorflow as tf

from config import flags
from model import Model
from utils.data import DataLoader

np.set_printoptions(precision=3)

# create log
log_dir = os.path.join(os.path.dirname(__file__), 'logs', flags.name)
os.makedirs('logs', exist_ok=True)
if os.path.exists(log_dir):
  shutil.rmtree(log_dir)
os.makedirs(os.path.join('logs', flags.name), exist_ok=True)
for fn in os.listdir('.'):
  if '.py' in fn:
    shutil.copy(fn, os.path.join(log_dir, fn))

f = open(os.path.join(log_dir, 'command.txt'), 'w')
f.write(' '.join(sys.argv) + '\n')
f.close()

# load data
dataloader = DataLoader(flags, data_dir='../data', obj_list=[3])

# create visualizer
if flags.viz:
  from utils.viz_util import Visualizer
  visualizer = Visualizer()

# create model
model = Model(flags, [dataloader.z_min, dataloader.z_max])

# create session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(log_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=0)

# load from checkpoint
if flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'models', flags.restore_name, '%04d.ckpt'%(flags.restore_epoch)))

print('start training ...')

# train
global_step = 0
for epoch in range(flags.epochs):
  batch_i = 0
  total_len = int(dataloader.min_data_size * len(dataloader.obj_list) // flags.batch_size)
  for obj_id, item_id, obs_hand, obs_obj in dataloader.fetch():
    batch_i += 1
    Z = np.random.random([flags.batch_size, flags.n_latent_factor])
    # Generate proposal with G
    gen_hand = sess.run(model.gen_hand, feed_dict={
      model.obs_obj: obs_obj, model.Z: Z, model.is_training: True
    })
    syn_hand = gen_hand.copy()
    energies = []
    # Update proposal with D
    for langevin_step in range(min(epoch * 5, flags.langevin_steps)):
      syn_hand, syn_energy, g_abs, g_ema = sess.run(model.langevin_result, feed_dict={
        model.syn_hand: syn_hand, model.obj_id: obj_id, model.is_training: True
      })
      energies.append(np.mean(syn_energy))
      # print()
      # print('g_abs', g_abs)
      # print('g_ema', g_ema)
    # Train G and D
    if epoch == 0:
      OE, GE, SE, GL, DL, _, _ = sess.run([
        model.obs_energy, model.gen_energy, model.syn_energy, 
        model.gen_loss, model.des_loss, model.train_gen, model.train_des
      ], feed_dict={
        model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, model.obj_id: obj_id, model.Z: Z, model.is_training:True
      })
      global_step += 1    
    else:
      OE, GE, SE, GL, DL, _, _, summary = sess.run([
        model.obs_energy, model.gen_energy, model.syn_energy, 
        model.gen_loss, model.des_loss, model.train_gen, model.train_des, model.summaries
      ], feed_dict={
        model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, model.obj_id: obj_id, model.Z: Z, model.is_training:True
      })
      train_writer.add_summary(summary, global_step=global_step)
      global_step += 1
    print('\r%d: %d/%d G:%f D:%f Improved Energy: %f'%(epoch, batch_i, total_len, GL, DL, np.mean(GE-SE)), end='')
    if flags.debug and global_step % 10 == 9:
      break
  print()
  # visualize
  # if flags.viz:
  #   for item in range(len(syn_hand)):
  #     visualizer.visualize_distance(obj_id, gen_hand[item], os.path.join(log_dir, 'epoch-%04d-gen-%d'%(epoch, item)))
  #     visualizer.visualize_distance(obj_id, syn_hand[item], os.path.join(log_dir, 'epoch-%04d-syn-%d'%(epoch, item)))
  if epoch > 0:
    saver.save(sess, os.path.join(log_dir, '%04d.ckpt'%epoch))
    pickle.dump([obj_id, gen_hand, syn_hand, GE, SE, g_ema], open(os.path.join(log_dir, '%04d.pkl'%epoch), 'wb'))
