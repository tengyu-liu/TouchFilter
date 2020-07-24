import os
import pickle
import shutil
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import flags
from gan_style_model import Model
from utils.data import DataLoader

def floatRgb(mag):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    cmax = np.max(mag)
    cmin = np.min(mag)
    x = (mag-cmin)/(cmax-cmin)
    x[np.isnan(x)] = 0.5
    x[np.isinf(x)] = 0.5
    blue  = (np.minimum(np.maximum(4*(0.75-x), 0.), 1.) * 255).astype(np.int32)
    red   = (np.minimum(np.maximum(4*(x-0.25), 0.), 1.) * 255).astype(np.int32)
    green = (np.minimum(np.maximum(4*np.fabs(x-0.5)-1., 0.), 1.) * 255).astype(np.int32)
    return np.stack([red, green, blue], axis=-1)
  
colors = floatRgb(np.linspace(0,15,16))

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
dataloader = DataLoader(flags, data_dir='../data', obj_list=[1,2,3,4,5,6,7], debug=flags.debug)

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

def get_lr(old_dl, old_gl):
  try:
    f = open('config.txt')
    dl, gl = [float(x.strip()) for x in f.read().split(',')]
    if dl != old_dl or gl != old_gl:
      print('\nUpdating learning rate to (%e, %e)'%(dl, gl))
    return dl, gl
  except:
    return old_dl, old_gl

cur_dl, cur_gl = flags.lr_des, flags.lr_gen
f = open('config.txt', 'w')
f.write('%e,%e'%(cur_dl, cur_gl))
f.close()

plt.ion()

# train
epoch = -1
# for epoch in range(flags.restore_epoch+1, flags.epochs):
while True:
  epoch += 1
  batch_i = 0
  total_len = int(dataloader.min_data_size * len(dataloader.obj_list) // flags.batch_size)
  for obj_id, item_id, obs_hand, obs_z, obs_obj, obj_trans, obj_rot, obs_idx  in dataloader.fetch():
    global_step += 1
    batch_i += 1
    # check for new learning rate instruction
    cur_dl, cur_gl = get_lr(cur_dl, cur_gl)
    syn_z = np.random.normal(loc=0, scale=1, size=[flags.batch_size, flags.n_latent_factor])
    syn_z /= np.linalg.norm(syn_z, axis=-1, keepdims=True)
    # Generate proposal with G
    # train D 
    DL, OE, GE, _, QC = sess.run([model.des_loss, model.obs_energy, model.gen_energy, model.train_des, model.qualified_candidates], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: False,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id,
      model.obs_hand: obs_hand,
      model.lr_des: cur_dl
    })
    # train G
    GL, GE2, _ = sess.run([model.gen_loss, model.gen_energy, model.train_gen], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: False,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id,
      model.lr_gen: cur_gl
    }) 
    if batch_i % 100 == 0:
      # check improvement
      OE2, GE3, obs_contact, gen_contact = sess.run([model.obs_energy, model.gen_energy, model.obs_contact, model.gen_contact], feed_dict={
        model.obs_obj: obs_obj,
        model.syn_z: syn_z,
        model.is_training: False,
        model.obs_obj_rot: obj_rot,
        model.obs_obj_trans: obj_trans,
        model.obj_id: obj_id,
        model.obs_hand: obs_hand
      })
      # write summary
      summary = sess.run(model.summaries, feed_dict={
        model.summ_oe: OE, model.summ_oe2: OE2, model.summ_ge: GE, model.summ_ge2: GE2, model.summ_ge3: GE3
      })
      train_writer.add_summary(summary, global_step=global_step)

      obs_colors = np.zeros([obs_contact.shape[0], obs_contact.shape[1], 3]) + 200
      gen_colors = np.zeros([gen_contact.shape[0], gen_contact.shape[1], 3]) + 200
      for i in range(16):
        obs_colors[np.arange(obs_colors.shape[0]), np.argmax(obs_contact[:,:,i], axis=1)] = colors[i]
        gen_colors[np.arange(gen_colors.shape[0]), np.argmax(gen_contact[:,:,i], axis=1)] = colors[i]
        
      plt.clf()
      # plt.subplot(121)
      plt.hist(obs_contact.reshape([-1]), label='obs', log=True)
      # plt.subplot(122)
      plt.hist(gen_contact.reshape([-1]), label='gen', log=True)
      plt.legend()
      plt.pause(1e-5)

      draw_obj = obs_obj - np.mean(obs_obj, axis=1, keepdims=True)
      summary = sess.run(model.pts_summaries, feed_dict={
        model.summ_vert:draw_obj, 
        model.summ_obs_color:obs_colors, 
        model.summ_gen_color:gen_colors
      })
      train_writer.add_summary(summary, global_step=global_step)
      
    print('\r%d: %d/%d G:%f D:%f'%(   
      epoch, batch_i, total_len, GL, DL), end='')

    # if flags.debug and global_step % 10 == 9:
    #   saver.save(sess, os.path.join(log_dir, '%04d.ckpt'%epoch))
    #   exit()

  print()
  # visualize
  # if flags.viz:
  #   for item in range(len(syn_hand)):
  #     visualizer.visualize_distance(obj_id, gen_hand[item], os.path.join(log_dir, 'epoch-%04d-gen-%d'%(epoch, item)))
  #     visualizer.visualize_distance(obj_id, syn_hand[item], os.path.join(log_dir, 'epoch-%04d-syn-%d'%(epoch, item)))
  if epoch > -1 and (not flags.debug or epoch % 100 == 0):
    saver.save(sess, os.path.join(log_dir, '%04d.ckpt'%epoch))
    gen_contact, GE, OE = sess.run([model.gen_contact, model.gen_energy, model.obs_energy], feed_dict={
      model.obs_obj: obs_obj,
      model.syn_z: syn_z,
      model.is_training: False,
      model.obs_obj_rot: obj_rot,
      model.obs_obj_trans: obj_trans,
      model.obj_id: obj_id,
      model.obs_hand: obs_hand,
    })
    pickle.dump([obj_id, gen_contact, obs_obj, obs_hand, GE, OE, obj_rot, obj_trans], open(os.path.join(log_dir, '%04d.pkl'%epoch), 'wb'))

