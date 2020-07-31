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

np.seterr(all='raise')
# np.random.seed(0)
# torch.manual_seed(0)

from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

# prepare argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--n_iter', default=1000000, type=int)
parser.add_argument('--annealing_period', default=100, type=int)
parser.add_argument('--starting_temperature', default=1, type=float)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--stepsize_period', default=100, type=int)
parser.add_argument('--noise_size', default=0.1, type=float)
parser.add_argument('--name', default='exp', type=str)
parser.add_argument('--mano_path', default='third_party/manopth/mano/models', type=str)
parser.add_argument('--viz', action='store_true')

args = parser.parse_args()

if args.viz:
  import matplotlib.pyplot as plt
  import plotly
  import plotly.graph_objects as go
  from mpl_toolkits.mplot3d import Axes3D

  plt.ion()
  plt.title(args.name)
  ax1 = plt.subplot(221)
  ax2 = plt.subplot(222)
  ax3 = plt.subplot(223)
  ax4 = plt.subplot(224, projection='3d')

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
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# get obj_code
obj_code, obj_idx = get_obj_code_random(args.batch_size)

def compute_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(args.batch_size), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)

  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(verts=hand_verts)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(3)], dim=1)
  hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
  normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
  surface_distance = fc_loss_model.dist_loss(obj_code, contact_point)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-args.n_handcode:], dim=-1)
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment


def compute_contact_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(args.batch_size), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  surface_distance = fc_loss_model.dist_loss(obj_code, contact_point)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-args.n_handcode:], dim=-1)
  if verbose:
    return surface_distance.sum(1), penetration.sum(1), z_norm
  else:
    return surface_distance.sum(1) + penetration.sum(1) + z_norm


def compute_closure_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(args.batch_size), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)

  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(verts=hand_verts)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(3)], dim=1)
  hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
  normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
  if verbose:
    return linear_independence, force_closure, normal_alignment
  else:
    return linear_independence + force_closure + normal_alignment


starting_temperature = torch.tensor(args.starting_temperature).float().cuda()
temperature_decay = torch.tensor(args.temperature_decay).float().cuda()
annealing_period = torch.tensor(args.annealing_period).float().cuda()
noise_size = torch.tensor(args.noise_size).float().cuda()
temperature_decay = torch.tensor(args.temperature_decay).float().cuda()
stepsize_period = torch.tensor(args.stepsize_period).float().cuda()

def T(t):
  # annealing schedule
  return starting_temperature * temperature_decay ** (t // annealing_period)

def S(t):
  return noise_size * temperature_decay ** (t // stepsize_period)

# initialize z and contact point
contact_point_indices = torch.randint(0, hand_model.num_points, size=[args.batch_size, 3], device='cuda')
z = torch.normal(mean=0, std=1, size=[args.batch_size, hand_model.code_length], requires_grad=True).float().cuda()

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

energy = compute_energy(obj_code, z, contact_point_indices)
grad = torch.autograd.grad(energy.sum(), z)[0]
energy_history = []
temperature_history = []
stepsize_history = []

grad_ema = EMA(args.mu)
grad_ema.apply(grad)

# mcmc
for _iter in range(args.n_iter):
  # 50/50 chance of updating z or contact point
  rand = random.random()
  step_size = S(_iter)
  temperature = T(_iter)
  if rand < 0.5:
    # update z using matropolis-hasting langevin
    noise = torch.normal(mean=0, std=1, size=z.shape, device='cuda').float() * step_size
    new_z = z - 0.5 * grad / grad_ema.average.unsqueeze(0) * step_size * step_size + noise
    new_contact_point_indices = contact_point_indices
  else:
    # update contact point
    update_indices = torch.randint(0, 3, size=[args.batch_size], device='cuda')
    update_to = torch.randint(0, hand_model.num_points, size=[args.batch_size], device='cuda')
    new_contact_point_indices = contact_point_indices.clone()
    new_contact_point_indices[torch.arange(args.batch_size), update_indices] = update_to
    new_z = z
  # compute new energy
  new_energy = compute_energy(obj_code, new_z, new_contact_point_indices)
  new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
  with torch.no_grad():
    # metropolis-hasting
    alpha = torch.rand(args.batch_size, device='cuda').float()
    accept = alpha < torch.exp((energy - new_energy) / temperature)
    z[accept] = new_z[accept]
    contact_point_indices[accept] = new_contact_point_indices[accept]
    energy[accept] = new_energy[accept]
    grad[accept] = new_grad[accept]
    grad_ema.apply(grad)

  if _iter % 100 == 0:
    print('\r%d: %f'%(_iter, energy.mean().detach().cpu().numpy()), end='', flush=True)
    energy_history.append(energy.mean().detach().cpu().numpy())
    temperature_history.append(temperature)
    stepsize_history.append(step_size)
    
    if args.viz:
      i_item = random.randint(0, args.batch_size-1)
      mesh = get_obj_mesh(obj_idx[i_item])
      ax1.cla()
      ax1.plot(energy_history)
      ax1.set_yscale('log')
      ax2.cla()
      ax2.plot(temperature_history)
      ax2.set_yscale('log')
      ax3.cla()
      ax3.plot(stepsize_history)
      ax3.set_yscale('log')
      ax4.cla()
      verts = hand_model.get_vertices(z).detach().cpu().numpy()
      cpi = contact_point_indices[i_item].detach().cpu().numpy()
      ax4.scatter(verts[i_item, :,0], verts[i_item, :,1], verts[i_item, :,2], color='lightpink', s=2)
      ax4.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], color='lightblue', s=2)
      ax4.scatter(verts[i_item, cpi, 0], verts[i_item, cpi, 1], verts[i_item, cpi, 2], color='red', s=5)
      ax4.set_xlim([-1,1])
      ax4.set_ylim([-1,1])
      ax4.set_zlim([-1,1])
      ax4.axis('off')

      plt.title(args.name)
      plt.pause(1e-5)

  if _iter % 2000 == 0:
    pickle.dump([obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history], open(os.path.join(log_dir, 'saved_%d.pkl'%_iter), 'wb'))
    print()
