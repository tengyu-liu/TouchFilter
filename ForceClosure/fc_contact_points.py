import sys
import shutil
import argparse

import torch
import torch.nn as nn
import torch.utils.tensorboard
import os
import numpy as np
import tensorboard
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HandModel import HandModel
from ObjectModel import ObjectModel
from Losses import FCLoss
from FCPrediction import FCPrediction
from PenetrationModel import PenetrationModel
from CodeUtil import *

# prepare argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_graspcode', default=10, type=int)
parser.add_argument('--n_iter', default=int(1e8), type=int)
parser.add_argument('--name', default='exp', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

log_dir = os.path.join('logs', args.name)
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'src'), exist_ok=True)
for fn in os.listdir('.'):
  if fn[-3:] == '.py':
    shutil.copy(fn, os.path.join(log_dir, 'src', fn))
f = open(os.path.join(log_dir, 'command.txt'), 'w')
f.write(' '.join(sys.argv))
f.close()

if args.viz:
  import matplotlib.pyplot as plt
  plt.ion()

# prepare models

object_model = ObjectModel(
  state_dict_path="/media/tengyu/2THDD/DeepSDF/DeepSDF/experiments/DeepSDF_antelope/ModelParameters/2000.pth"
)

fc_loss = FCLoss(object_model=object_model)
fc_prediction = FCPrediction().cuda().train()

# prepare utilities
optimizer = torch.optim.Adam(
  fc_prediction.parameters(),
  lr=args.lr, betas=(0.99, 0.999), eps=1e-8)

writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)

def debug(msg=''):
  if args.debug:
    print(msg)

"""
FIXME: Error summary
It seems that the gradient of the normal is pushing x outward 
and to a same but random direction. Not sure why this is the case. 
"""
ax = plt.subplot(111, projection='3d')
for _iter in range(args.n_iter):
  # Load object code
  obj_code = get_obj_code_random(args.batch_size)
  grasp_code = get_grasp_code_random(args.batch_size, args.n_graspcode)
  # predict contact point pairs and friction
  cp_o = fc_prediction(obj_code, grasp_code)
  # find distance and normal
  cp_o_dist = object_model.distance(obj_code, cp_o)
  cp_o_normal = object_model.gradient(cp_o, cp_o_dist, retain_graph=True, create_graph=True)
  cp_o_normal = cp_o_normal / torch.norm(cp_o_normal, dim=-1, keepdim=True)
  if args.viz and _iter % 200 == 0:
    # visualize everything
    obj_pts = torch.rand(size=[1, 2000, 3], requires_grad=True).float().cuda() * 2 - 1
    obj_pts = object_model.closest_point(obj_code[[0]], obj_pts)[0].detach().cpu().numpy()
    ax.cla()
    ax.scatter(obj_pts[0,:,0], obj_pts[0,:,1], obj_pts[0,:,2], s=2, c='blue')
    ax.scatter(cp_o.detach().cpu().numpy()[0,:,0], cp_o.detach().cpu().numpy()[0,:,1], cp_o.detach().cpu().numpy()[0,:,2], s=20, c='red')
    ax.quiver(
      cp_o.detach().cpu().numpy()[0,:,0], 
      cp_o.detach().cpu().numpy()[0,:,1], 
      cp_o.detach().cpu().numpy()[0,:,2], 
      cp_o_normal.detach().cpu().numpy()[0,:,0], 
      cp_o_normal.detach().cpu().numpy()[0,:,1], 
      cp_o_normal.detach().cpu().numpy()[0,:,2], 
      length=0.1, normalize=True, color='green')
    # cx, cy, cz = obj_pts[0].mean(0)
    # ax.set_xlim([cx-1, cx+1])
    # ax.set_ylim([cy-1, cy+1])
    # ax.set_zlim([cz-1, cz+1])
    # ax.axis('off')
  plt.pause(1e-5)
  # compute losses
  fca, fcb, fcd = map(torch.mean, fc_loss.fc_loss(cp_o, cp_o_normal, obj_code))  # FIXME: changes made: use cp_o_normal as friction
  loss = fca + fcb + fcd
  # update weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  # write summary
  writer.add_scalar('Loss/ForceClosure_8a', fca.detach(), _iter)
  writer.add_scalar('Loss/ForceClosure_8b', fcb.detach(), _iter)
  writer.add_scalar('Loss/ForceClosure_8d', fcd.detach(), _iter)
  writer.add_scalar('Loss/Total', loss.detach(), _iter)
  print('\riter: %d loss: %f, %f, %f'%(_iter, fca, fcb, fcd), end='')
  
  if (_iter+1) % 1000 == 0:
    # save model
    torch.save(fc_prediction.state_dict(), os.path.join(log_dir, 'weights', '%d.pth'%_iter))
    print()