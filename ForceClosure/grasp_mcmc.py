import argparse
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.utils.tensorboard

from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from HCGraspPrediction import GraspPrediction
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

np.seterr(all='raise')

# prepare argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--n_iter', default=1000000, type=int)
parser.add_argument('--annealing_period', default=100, type=int)
parser.add_argument('--starting_temperature', default=1, type=float)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--stepsize_period', default=10000, type=int)
parser.add_argument('--update_size', default=0.1, type=float)
parser.add_argument('--name', default='exp', type=str)
parser.add_argument('--mano_path', default='/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/ForceClosure/third_party/manopth/mano/models', type=str)
parser.add_argument('--viz', action='store_true')

args = parser.parse_args()

if args.viz:
  import matplotlib.pyplot as plt
  import plotly
  import plotly.graph_objects as go
  from mpl_toolkits.mplot3d import Axes3D

  plt.ion()
  plt.title(args.name)
  ax1 = plt.subplot(131)
  ax2 = plt.subplot(132)
  ax3 = plt.subplot(133)

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

# prepare models
hand_model = HandModel(
  n_handcode=args.n_handcode,
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False,
  mano_path=args.mano_path)

hand_verts_eye = torch.tensor(np.eye(hand_model.num_points)).float().cuda() # 778 x 778

object_model = ObjectModel(
  state_dict_path="data/ModelParameters/2000.pth"
)

fc_loss_model = FCLoss(object_model=object_model)
grasp_prediction = GraspPrediction(num_cp=3, hand_code_length=hand_model.code_length, num_handpoint=hand_model.num_points).cuda()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# get obj_code
obj_code, obj_idx = get_obj_code_random(args.batch_size)

def compute_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(args.batch_size), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)

  with torch.no_grad():
    hand_normal = hand_model.get_surface_normals(verts=hand_verts)
    closest_distances, closest_indices = torch.norm(hand_verts.unsqueeze(2) - contact_point.unsqueeze(1), dim=-1).min(1)
    closest_normals = torch.stack([hand_normal[torch.arange(args.batch_size), closest_indices[:,i], :] for i in range(3)], dim=1)
    closest_normals = closest_normals / torch.norm(closest_normals, dim=-1, keepdim=True)    
    hand_loss = closest_distances.sum(1)
    normal_alignment = ((closest_normals * contact_normal).sum(-1) + 1).sum(-1)
    linear_independence, force_closure, surface_distance = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
    penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
    z_norm = torch.norm(z[:,-args.n_handcode:], dim=-1)
    loss = hand_loss * 0.1 + linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm * 0.1 + normal_alignment
    if verbose:
      return hand_loss * 0.1, linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm * 0.1, normal_alignment
    else:
      return loss

def T(t):
  # annealing schedule
  return args.starting_temperature * args.temperature_decay ** (t // args.annealing_period)

def S(t):
  return args.update_size * 0.95 ** (t // args.stepsize_period)

# initialize z and contact point
contact_point_indices = torch.randint(0, hand_model.num_points, size=[args.batch_size, 3]).cuda()
z = torch.normal(mean=0, std=1, size=[args.batch_size, hand_model.code_length], requires_grad=True).float().cuda()
energy = compute_energy(obj_code, z, contact_point_indices)
energy_history = []
temperature_history = []
stepsize_history = []

# align palm facing direction
for align_iter in range(3):
  hand_verts = hand_model.get_vertices(z)
  back_direction = hand_model.back_direction(verts=hand_verts)
  palm_point = hand_verts[:, [hand_model.facedir_base_ids[1]], :]
  palm_normal = object_model.gradient(palm_point, object_model.distance(obj_code, palm_point))[:,0,:]
  z = hand_model.align(z, back_direction, palm_normal)

with torch.no_grad():
# backup from penetration
  back_direction = hand_model.back_direction(z)
  for penetration_iter in range(3):
    max_penetration = penetration_model.get_max_penetration(obj_code, z).unsqueeze(1)
    z = torch.cat([z[:,:3] + back_direction * max_penetration, z[:,3:]], dim=-1)

z.requires_grad_()

# mcmc
for _iter in range(args.n_iter):
  # 50/50 chance of updating z or contact point
  rand = random.random()
  if rand < 0.5:
    # update z
    z_update = torch.normal(mean=0, std=1, size=z.shape).float().cuda() * S(_iter) * torch.randint(0, 2, size=z.shape).float().cuda()
    new_z = z + z_update
    new_contact_point_indices = contact_point_indices
  else:
    # update contact point
    update_indices = torch.randint(0, 3, size=[args.batch_size]).cuda()
    update_to = torch.randint(0, hand_model.num_points, size=[args.batch_size]).cuda()
    new_contact_point_indices = contact_point_indices.clone()
    new_contact_point_indices[torch.arange(args.batch_size), update_indices] = update_to
    new_z = z
  # compute new energy
  new_energy = compute_energy(obj_code, new_z, new_contact_point_indices)
  with torch.no_grad():
    # metropolis-hasting
    alpha = torch.rand(args.batch_size).float().cuda()
    accept = alpha < torch.exp((energy - new_energy) / T(_iter))
    z[accept] = new_z[accept]
    contact_point_indices[accept] = new_contact_point_indices[accept]
    energy[accept] = new_energy[accept]

  print('\r%d: %f'%(_iter, energy.mean().detach().cpu().numpy()), end='', flush=True)
  energy_history.append(energy.mean().detach().cpu().numpy())
  temperature_history.append(T(_iter))
  stepsize_history.append(S(_iter))

  if _iter % 10 == 0 and args.viz:
    ax1.cla()
    ax1.plot(energy_history)
    ax1.set_yscale('log')
    ax2.cla()
    ax2.plot(temperature_history)
    ax2.set_yscale('log')
    ax3.cla()
    ax3.plot(stepsize_history)
    ax3.set_yscale('log')

    plt.title(args.name)
    plt.pause(1e-5)

  if _iter % 1000 == 0:
    pickle.dump([obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history], open(os.path.join(log_dir, 'saved_%d.pkl'%_iter), 'wb'))

