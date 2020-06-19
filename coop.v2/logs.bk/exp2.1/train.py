import os
import pickle
import shutil
import sys
import random

import numpy as np
import tensorflow as tf

from config import flags
from model import Model
from utils.data import DataLoader

np.set_printoptions(precision=3)

np.random.seed(0)
tf.random.set_random_seed(0)
random.seed(0)

# create log
log_dir = os.path.join(os.path.dirname(__file__), 'logs', flags.name)
os.makedirs('logs', exist_ok=True)
if os.path.exists(log_dir):
  shutil.rmtree(log_dir)
os.makedirs(os.path.join('logs', flags.name), exist_ok=True)
shutil.copytree('utils', os.path.join(log_dir, 'utils'))
for fn in os.listdir('.'):
  if '.py' in fn:
    shutil.copy(fn, os.path.join(log_dir, fn))

f = open(os.path.join(log_dir, 'command.txt'), 'w')
f.write(' '.join(sys.argv) + '\n')
f.close()

# load data
dataloader = DataLoader(flags, data_dir='../data', obj_list=[1,2,3,4,5,6,7])

# create visualizer
if flags.viz:
  from utils.viz_util import Visualizer
  import matplotlib.pyplot as plt
  plt.ion()
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
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'logs', flags.restore_name, '%04d.ckpt'%(flags.restore_epoch)))
    dataloader.restore(os.path.join(log_dir, '%04d.pkl'%epoch))

print('start training ...')



# train
global_step = 0
generator_loss_improvement = []
descriptor_loss_improvement = []
generator_loss_improvement_moving_average = []
descriptor_loss_improvement_moving_average = []
for epoch in range(flags.restore_epoch+1, flags.epochs):
  batch_i = 0
  total_len = int(dataloader.min_data_size * len(dataloader.obj_list) // flags.batch_size)
  cumulative_qualification = 0
  cumulative_total = 0
  for obj_id, item_id, obs_hand, obs_z, obs_obj, obj_trans, obj_rot, obs_idx  in dataloader.fetch():
    batch_i += 1
    syn_z = np.random.normal(loc=0, scale=1, size=[flags.batch_size, flags.n_latent_factor])
    syn_z /= np.linalg.norm(syn_z, axis=-1, keepdims=True)
    obs_z /= np.linalg.norm(obs_z, axis=-1, keepdims=True)
    obs_z_backup = obs_z.copy()
    # Generate proposal with G
    gen_hand = sess.run(model.gen_hand, feed_dict={
      model.obs_obj: obs_obj, model.syn_z: syn_z, model.is_training: True, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans
    })
    syn_hand = gen_hand.copy()
    energies = []
    # Update proposal with D
    for langevin_step in range(flags.langevin_steps):
      syn_hand, _, syn_energy, g_abs, g_ema = sess.run(model.langevin_result, feed_dict={
        model.syn_hand: syn_hand, model.obj_id: obj_id, model.is_training: True, model.syn_z: syn_z, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans
      })
      energies.append(syn_energy)
      # print()
      # print('g_abs', g_abs)
      # print('g_ema', g_ema)
    # Train G and D
    
    OE, OC, GE, GC, SE, SC = sess.run([model.obs_energy, model.obs_contact, model.gen_energy, model.gen_contact, model.syn_energy, model.syn_contact], feed_dict={
      model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, model.obj_id: obj_id, model.syn_z: syn_z, 
      model.obs_z: obs_z, model.is_training:True, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans
    })

    bad_synthesis = SE > GE
    syn_hand[bad_synthesis] = gen_hand[bad_synthesis]
    SC[bad_synthesis] = GC[bad_synthesis]
    SE[bad_synthesis] = GE[bad_synthesis]

    energies.append(SE)

    qualified_candidates = np.expand_dims(SE <= np.mean(OE), axis=-1)

    cumulative_qualification += sum(qualified_candidates)
    cumulative_total += len(qualified_candidates)

    GL = np.nan
    DL = np.nan
    DL2 = np.nan

    if sum(qualified_candidates) > 0:
      GL, DL, _, _, summary = sess.run([
        model.gen_loss, model.des_loss, model.train_gen, model.train_des, model.summaries
      ], feed_dict={
        model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, 
        model.obj_id: obj_id, model.syn_z: syn_z, model.obs_z: obs_z, model.is_training:True, 
        model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans, model.qualified_candidates: qualified_candidates
      })

      GL2, DL2 = sess.run([model.gen_loss, model.des_loss], feed_dict={
          model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, 
          model.obj_id: obj_id, model.syn_z: syn_z, model.obs_z: obs_z, model.is_training:True, 
          model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans, model.qualified_candidates: qualified_candidates
      })
      descriptor_loss_improvement.append(DL-DL2)

      for langevin_step in range(1):
        _, obs_z, _, _, _ = sess.run(model.langevin_result, feed_dict={
          model.syn_hand: obs_hand, model.obj_id: obj_id, model.is_training: True, model.syn_z: obs_z, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans
        })
        obs_z /= np.linalg.norm(obs_z, axis=-1, keepdims=True)
      dataloader.update_z(obj_id, obs_z, obs_idx)

      train_writer.add_summary(summary, global_step=global_step)
      global_step += 1
    else:
      GL, _, summary = sess.run([model.gen_loss, model.train_gen, model.summaries], feed_dict={
        model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, 
        model.obj_id: obj_id, model.syn_z: syn_z, model.obs_z: obs_z, model.is_training:True, 
        model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans, model.qualified_candidates: qualified_candidates
      })

      GL2 = sess.run(model.gen_loss, feed_dict={
          model.obs_obj: obs_obj, model.obs_hand: obs_hand, model.syn_hand:syn_hand, model.syn_z: syn_z, model.is_training:True
      })

    generator_loss_improvement.append(GL-GL2)

    generator_loss_improvement_moving_average.append(np.mean(generator_loss_improvement))
    descriptor_loss_improvement_moving_average.append(np.mean(descriptor_loss_improvement))

    if len(generator_loss_improvement) > 100:
      generator_loss_improvement.pop(0)
    if len(descriptor_loss_improvement) > 100:
      descriptor_loss_improvement.pop(0)
    if len(generator_loss_improvement_moving_average) > 1000:
      generator_loss_improvement_moving_average.pop(0)
    if len(descriptor_loss_improvement_moving_average) > 1000:
      descriptor_loss_improvement_moving_average.pop(0)

    if flags.viz and batch_i % 10 == 9:
      plt.clf()
      plt.subplot(321)
      energies = np.array(energies)
      plt.scatter(np.zeros_like(OE)-1, OE, c='red', s=2)
      for i in range(energies.shape[1]):
        if qualified_candidates[i]:
          plt.plot(energies[:,i])
        else:
          plt.plot(energies[:,i], linestyle='dashed')

      plt.subplot(323)
      plt.hist(generator_loss_improvement, bins=100)
      if len(generator_loss_improvement) > 80:
        plt.subplot(324)
        plt.plot(generator_loss_improvement_moving_average)
        plt.plot(np.zeros_like(generator_loss_improvement_moving_average))
      plt.subplot(325)
      plt.hist(descriptor_loss_improvement, bins=100)
      if len(descriptor_loss_improvement) > 80:
        plt.subplot(326)
        plt.plot(descriptor_loss_improvement_moving_average)
        plt.plot(np.zeros_like(descriptor_loss_improvement_moving_average))
      plt.pause(1e-5)

    print('\r%d: %d/%d cum.qual.:%.2f%% G:%f D:%f G.I.:%f(%f) D.I.: %f(%f) Improved Energy: %f'%(
      epoch, batch_i, total_len, 100.*cumulative_qualification/cumulative_total, GL, DL, 
      GL-GL2, generator_loss_improvement_moving_average[-1], 
      DL-DL2, descriptor_loss_improvement_moving_average[-1], np.mean(GE-SE)), end='')

    if flags.debug and global_step % 10 == 9:
      saver.save(sess, os.path.join(log_dir, '%04d.ckpt'%epoch))
      exit()

  print()
  # visualize
  # if flags.viz:
  #   for item in range(len(syn_hand)):
  #     visualizer.visualize_distance(obj_id, gen_hand[item], os.path.join(log_dir, 'epoch-%04d-gen-%d'%(epoch, item)))
  #     visualizer.visualize_distance(obj_id, syn_hand[item], os.path.join(log_dir, 'epoch-%04d-syn-%d'%(epoch, item)))
  if epoch > -1:
    saver.save(sess, os.path.join(log_dir, '%04d.ckpt'%epoch))
    pickle.dump([obj_id, gen_hand, GC, syn_hand, SC, obs_hand, OC, GE, SE, OE, g_ema, dataloader.obs_z2s, obj_rot, obj_trans], open(os.path.join(log_dir, '%04d.pkl'%epoch), 'wb'))

# cum.qual. 0.69% @ 145 batch - gen=1e-4
# cum.qual. 0.85% @ 147 batch - gen=1e-3
# cum.qual. xxxx% @ 145 batch - gen=1e-3 & only update obs_z if there is qualified candidates
# TODO: move obs_z update to qual block