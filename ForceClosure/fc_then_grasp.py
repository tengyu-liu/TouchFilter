from EMA import EMA
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

batch_size = 10

object_model = ObjectModel(
  state_dict_path="/media/tengyu/2THDD/DeepSDF/DeepSDF/experiments/DeepSDF_antelope/ModelParameters/2000.pth"
)

hand_model = HandModel(
  n_handcode=6,
  root_rot_mode='ortho6d', 
  robust_rot=True,
  mano_path='/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/ForceClosure/third_party/manopth/mano/models')

fc_loss = FCLoss(object_model)
fc_prediction = FCPrediction().cuda()
fc_prediction.load_state_dict(torch.load('logs/fc/weights/102999.pth'))
fc_prediction.train()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

obj_code = get_obj_code_random(batch_size=batch_size)

# get obj pointcloud
obj_pts = torch.tensor(np.random.random([batch_size,2000,3]), requires_grad=True).float().cuda() * 2 - 1
distance = object_model.distance(obj_code, obj_pts)
while distance.abs().mean() > 0.003:
  distance = object_model.distance(obj_code, obj_pts)
  gradient = object_model.gradient(obj_pts, distance)
  obj_pts = obj_pts - gradient * distance

obj_pts = obj_pts[0].detach().cpu().numpy()

grasp_code = get_grasp_code_random(batch_size=batch_size, code_length=10)
hand_code = torch.tensor(np.random.random([batch_size, hand_model.code_length])).float().cuda() * 0.1

plt.ion()
ax = plt.subplot(121, projection='3d')
ax2 = plt.subplot(122)

cp_o = fc_prediction(obj_code, grasp_code)

std = 0.01

def compute_energy(hand_code, obj_code, cp_o, cp_h=None):
  if cp_h is None:
    cp_h = hand_model.closest_point(hand_code, cp_o)
  penetration = penetration_model.get_penetration(obj_code, hand_code).sum(1).squeeze()
  difference = fc_loss.l2_norm(cp_h - cp_o).sum()
  energy = penetration + difference
  return energy.detach()

energy = compute_energy(hand_code, obj_code, cp_o)

energies = []
energies.append(energy.mean().cpu().numpy())

_iter = 0
while True:
  _iter += 1
  T = 10 / _iter 
  hand_verts = hand_model.get_vertices(hand_code)
  cp_h = hand_model.closest_point(hand_code, cp_o)
  
  hv = hand_verts[0].detach().cpu().numpy()

  h2o_dist = object_model.distance(obj_code, hand_verts)[0,:,0].detach().cpu().numpy()
  p = h2o_dist > 0
  n = h2o_dist <= 0

  cph = cp_h[0].detach().cpu().numpy()
  cpo = cp_o[0].detach().cpu().numpy()
  ax.cla()
  ax.scatter(hv[p,0], hv[p,1], hv[p,2], s=1, c='lightpink')
  ax.scatter(hv[n,0], hv[n,1], hv[n,2], s=20, c='black')
  ax.scatter(obj_pts[:,0], obj_pts[:,1], obj_pts[:,2], s=1, c='lightblue')
  ax.scatter(cpo[:,0], cpo[:,1], cpo[:,2], s=10, c='black')
  ax.axis('off')
  cx, cy, cz = (obj_pts.mean(0) + hv.mean(0))/2
  ax.set_xlim([cx-1, cx+1])
  ax.set_ylim([cy-1, cy+1])
  ax.set_zlim([cz-1, cz+1])
  
  mask = torch.rand(hand_code.shape).float().cuda() < 0.1
  new_hand_code = hand_code + torch.normal(mean=0, std=std, size=hand_code.shape).float().cuda() * mask
  new_energy = compute_energy(new_hand_code, obj_code, cp_o)

  delta_energy = (energy - new_energy) / T
  p = torch.exp(delta_energy)
  alpha = torch.rand(batch_size).float().cuda()
  
  hand_code[alpha < p] = new_hand_code[alpha < p]
  energy[alpha < p] = new_energy[alpha < p]
  
  energies.append(energy.mean().cpu().numpy())
  ax2.cla()
  ax2.plot(energies)

  plt.pause(1e-5)