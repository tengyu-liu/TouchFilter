import argparse
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import DefaultDict, Tuple

import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.utils.tensorboard

from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

np.seterr(all='raise')

# parse arguments
parser = argparse.ArgumentParser()
# - model parameters
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--mano_path', default='third_party/manopth/mano/models', type=str)
# - ADELM parameters
parser.add_argument('--T', default=0.1, type=float)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--delta', default=0.15, type=float) # earth moving distance between contact points
parser.add_argument('--M', default=1000, type=int)
parser.add_argument('--mu', default=0.98, type=float)
# - MCMC parameters
parser.add_argument('--stepsize', default=0.1, type=float)
# - data loading
parser.add_argument('--data_path', default='logs/sample_0/optimized_998000.pkl', type=str)
args = parser.parse_args()

# prepare models
hand_model = HandModel(
  n_handcode=args.n_handcode,
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False,
  mano_path=args.mano_path)

object_model = ObjectModel(
  state_dict_path="data/ModelParameters/2000.pth"
)

grad_ema = EMA(args.mu)

fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# compute energy
def compute_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(z.shape[0]), contact_point_indices[:,i],:] for i in range(3)], dim=1)
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

def distance(X, Y):
  object_x, z_x, contact_point_indices_x = X
  object_y, z_y, contact_point_indices_y = Y
  return hand_model.manifold_distance(contact_point_indices_x, contact_point_indices_y)

def MCMC(X, Xstar, T, alpha):
  object_x, z_x, contact_point_indices_x = X
  object_star, z_star, contact_point_indices_star = Xstar
  d = distance(X, Xstar)
  energy = compute_energy(object_x, z_x, contact_point_indices_x)
  rand = random.random()
  if rand < 0.5:
    # update z
    grad = torch.autograd.grad(energy.sum(), z_x)[0]
    grad_ema.apply(grad)
    noise = torch.normal(mean=0, std=1, size=z_x.shape, device='cuda').float() * args.stepsize
    new_z = z_x - 0.5 * grad / grad_ema.average.unsqueeze(0) * args.stepsize * args.stepsize + noise
    new_contact_point_indices = contact_point_indices_x
    new_d = d
  else:
    # update contact point
    update_indices = torch.randint(0, 3, size=[z_x.shape[0]], device='cuda')
    update_to = torch.randint(0, hand_model.num_points, size=[z_x.shape[0]], device='cuda')
    new_contact_point_indices = contact_point_indices_x.clone()
    new_contact_point_indices[torch.arange(z_x.shape[0]), update_indices] = update_to
    new_z = z_x
  # compute new energy
  new_energy = compute_energy(object_x, new_z, new_contact_point_indices)
  new_d =  distance([object_x, new_z, new_contact_point_indices], [object_star, z_star, contact_point_indices_star])
  with torch.no_grad():
    # metropolis-hasting
    alpha = torch.rand(z_x.shape[0], device='cuda').float()
    accept = alpha < torch.exp((energy - new_energy * alpha * (d - new_d)) / T)
    z_x[accept] = new_z[accept]
    d[accept] = new_d[accept]
    contact_point_indices_x[accept] = new_contact_point_indices[accept]
    energy[accept] = new_energy[accept]
  return [object_x, z_x, contact_point_indices_x], d, energy

def attraction_diffusion(C, Xstar, alpha, T, M):
  d = distance(C, Xstar)
  dstar = d
  m = torch.zeros([d.shape[0]], device='cuda').float()
  B = None
  not_finished = torch.logical_and(d > alpha, m < M)
  while not_finished.sum() > 0:
    C, d, U = MCMC(C, Xstar, T, alpha)
    # update energy barrier
    if B is None:
      B = U
    else:
      B_update = torch.logical_and(not_finished, B < U)
      B[B_update] = U[B_update]
    d_update_1 = torch.logical_and(not_finished, d >= dstar)
    d_update_2 = torch.logical_and(not_finished, d < dstar)
    m[d_update_1] = m[d_update_1] + 1
    m[d_update_2] = 0
    dstar[d_update_2] = d[d_update_2]
    not_finished = torch.logical_and(d > alpha, m < M)
  return d, B

def load_proposals(path):
  obj_code, z, contact_point_indices, energy, _, _, _ = pickle.load(open(path, 'rb'))
  Y = list(zip(obj_code, z, contact_point_indices))
  # linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = energy
  # energy = linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm * 0.1 + normal_alignment
  return Y[:10], energy[:10].detach().cpu().numpy()

def tile(Y, size):
  obj_code, z, contact_point_indices = Y
  return obj_code.unsqueeze(0).repeat([size, 1]), z.unsqueeze(0).repeat([size, 1]), contact_point_indices.unsqueeze(0).repeat([size, 1])

def collate(Z):
  codes, zs, indices = [], [], []
  for obj_code, z, contact_point_indices in Z:
    codes.append(obj_code)
    zs.append(z)
    indices.append(contact_point_indices)
  return torch.stack(codes, 0), torch.stack(zs, 0), torch.stack(indices, 0)

examples, example_energies = load_proposals(args.data_path)  # each proposal is already a local minimum

# ADELM
basin_labels = []
basin_minima = []
basin_minima_energies = []
def inf():
  return float('inf')

energy_barriers: DefaultDict[Tuple[int, int], float] = defaultdict(inf)

for n in range(len(examples)):
  if n == 0:
    basin_minima.append(examples[n])
    basin_minima_energies.append(example_energies[n])
    basin_labels.append(0)
  else:
    number_of_basins = max(basin_labels) + 1
    candidate_labels = []
    barrier_to_basins = []

    Yn = tile(examples[n], len(basin_minima))
    Zs = collate(basin_minima)

    d1, B1 = attraction_diffusion(Yn, Zs, args.alpha, args.T, args.M)
    d2, B2 = attraction_diffusion(Zs, Yn, args.alpha, args.T, args.M)

    for j in range(len(basin_minima)):
      if min(d1[j], d2[j]) < args.delta:
        candidate_labels.append(j)
        if d1[j] > args.delta:
          barrier_to_basins.append(B2[j])
        elif d2[j] > args.delta:
          barrier_to_basins.append(B1[j])
        else:
          barrier_to_basins.append(min(B1[j], B2[j]))

    if len(candidate_labels) == 0:
      basin_minima.append(examples[n])
      basin_minima_energies.append(example_energies[n])
      basin_labels.append(number_of_basins)
    else:
      basin_labels.append(candidate_labels[np.argmin(barrier_to_basins)])
      if example_energies[n] < basin_minima_energies[basin_labels[n]]:
        basin_minima_energies[basin_labels[n]] = example_energies[n]
        basin_minima[basin_labels[n]] = examples[n]
    
    current_label = basin_labels[-1]
    for j in range(number_of_basins):
      energy_barriers[(current_label, j)] = min(energy_barriers[(current_label, j)], barrier_to_basins[j])
      energy_barriers[(j, current_label)] = min(energy_barriers[(current_label, j)], barrier_to_basins[j])

  print(n, basin_labels, basin_minima_energies)

pickle.dump([basin_labels, basin_minima, basin_minima_energies, energy_barriers], open('ADELM.pkl', 'wb'))
