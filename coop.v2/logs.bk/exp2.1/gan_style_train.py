import os
import pickle
import shutil
import sys
import random

import numpy as np
import tensorflow as tf

from config import flags
from gan_style_model import Model
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

global_step = 0

# load from checkpoint
if flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'logs', flags.restore_name, '%04d.ckpt'%(flags.restore_epoch)))
    dataloader.restore(os.path.join(os.path.dirname(__file__), 'logs', flags.restore_name, '%04d.pkl'%flags.restore_epoch))
    global_step = (flags.restore_epoch + 1) * 2703 
print('start training ...')



# train
for epoch in range(flags.restore_epoch+1, flags.epochs):
  batch_i = 0
  total_len = int(dataloader.min_data_size * len(dataloader.obj_list) // flags.batch_size)
  for obj_id, item_id, obs_hand, obs_z, obs_obj, obj_trans, obj_rot, obs_idx  in dataloader.fetch():
    global_step += 1
    batch_i += 1
    syn_z = np.random.normal(loc=0, scale=1, size=[flags.batch_size, flags.n_latent_factor])
    syn_z /= np.linalg.norm(syn_z, axis=-1, keepdims=True)
    obs_z /= np.linalg.norm(obs_z, axis=-1, keepdims=True)
    obs_z_backup = obs_z.copy()
    # Generate proposal with G
    # train D 
    DL, _, summary = sess.run([model.des_loss, model.train_des, model.summaries], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: True,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id,
      model.obs_hand: obs_hand,
      model.obs_z: obs_z
    })
    # # check improvement
    # OE2, DL2 = sess.run([model.obs_energy, model.des_loss], feed_dict={
    #   model.obs_obj: obs_obj,
    #   model.syn_z: syn_z,
    #   model.is_training: True,
    #   model.obs_obj_rot: obj_rot,
    #   model.obs_obj_trans: obj_trans,
    #   model.obj_id: obj_id,
    #   model.obs_hand: obs_hand,
    #   model.obs_z: obs_z
    # })
    # train G
    GL, _ = sess.run([model.gen_loss, model.train_gen], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: True,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id
    }) 
    # # check improvement
    # GE2, GL2, OE2, DL2,  = sess.run([model.gen_energy, model.gen_loss, model.obs_energy, model.des_loss, model.summaries], feed_dict={
    #   model.obs_obj: obs_obj,
    #   model.syn_z: syn_z,
    #   model.is_training: True,
    #   model.obs_obj_rot: obj_rot,
    #   model.obs_obj_trans: obj_trans,
    #   model.obj_id: obj_id,
    #   model.obs_hand: obs_hand,
    #   model.obs_z: obs_z
    # })
    # update obs_z
    obs_z, _ = sess.run(model.langevin_result, feed_dict={
      model.syn_hand: obs_hand, model.obj_id: obj_id, model.is_training: True, model.syn_z: obs_z, model.obs_obj_rot: obj_rot, model.obs_obj_trans: obj_trans
    })
    obs_z /= np.linalg.norm(obs_z, axis=-1, keepdims=True)
    dataloader.update_z(obj_id, obs_z, obs_idx)
    # write summary
    train_writer.add_summary(summary, global_step=global_step)

    print('\r%d: %d/%d G:%f D:%f'%(
      epoch, batch_i, total_len, GL, DL), end='')

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
    gen_hand, GC, GE, OC, OE = sess.run([model.gen_hand, model.gen_contact, model.gen_energy, model.obs_contact, model.obs_energy], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: True,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id,
      model.obs_hand: obs_hand,
      model.obs_z: obs_z
    })
    pickle.dump([obj_id, gen_hand, GC, obs_hand, OC, GE, OE, dataloader.obs_z2s, obj_rot, obj_trans], open(os.path.join(log_dir, '%04d.pkl'%epoch), 'wb'))

# cum.qual. 0.69% @ 145 batch - gen=1e-4
# cum.qual. 0.85% @ 147 batch - gen=1e-3
# cum.qual. xxxx% @ 145 batch - gen=1e-3 & only update obs_z if there is qualified candidates
# TODO: move obs_z update to qual block