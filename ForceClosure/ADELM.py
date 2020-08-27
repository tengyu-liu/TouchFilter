import argparse
import copy
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import DefaultDict, Tuple, Dict, List

import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.utils.tensorboard
import plotly 
import plotly.graph_objects as go

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
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--delta', default=0.15, type=float) # earth moving distance between contact points
parser.add_argument('--M', default=1000, type=int)
parser.add_argument('--mu', default=0.98, type=float)
# - MCMC parameters
parser.add_argument('--T', default=0.1, type=float)
parser.add_argument('--stepsize', default=0.1, type=float)
# - data loading
parser.add_argument('--data_path', default='logs/sample_0/optimized_998000.pkl', type=str)
parser.add_argument('--time', action='store_true')
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
  # hand_verts_x = hand_model.get_vertices(z_x)
  # hand_verts_y = hand_model.get_vertices(z_y)
  # sdx = object_model.distance(object_x, hand_verts_x).squeeze()
  # sdy = object_model.distance(object_y, hand_verts_y).squeeze()
  # px = 1 / torch.abs(sdx)
  # py = 1 / torch.abs(sdy)
  # px = px / px.sum(1, keepdim=True)
  # py = py / py.sum(1, keepdim=True)
  # distance = py * (torch.log(py) - torch.log(px))
  # return distance.sum(1)
  distance = hand_model.manifold_distance(contact_point_indices_x, contact_point_indices_y)
  return distance

def MCMC(X, Xstar, T, alpha, directly_update_contact_points_flag):
  object_x, z_x, contact_point_indices_x = X
  z_x.requires_grad_()
  object_star, z_star, contact_point_indices_star = Xstar
  d = distance(X, Xstar)
  energy = compute_energy(object_x, z_x, contact_point_indices_x)
  rand = random.random()
  directly_update_contact_points = directly_update_contact_points_flag > 1
  new_contact_point_indices = contact_point_indices_x.clone()
  if rand < 0.85:
    # update z
    grad = torch.autograd.grad((energy + d * alpha).sum(), z_x)[0]
    grad_ema.apply(grad)
    noise = torch.normal(mean=0, std=1, size=z_x.shape, device='cuda').float() * args.stepsize
    new_z = z_x - 0.5 * grad / grad_ema.average.unsqueeze(0) * args.stepsize * args.stepsize + noise
  else:
    # update contact point
    directly_update_contact_points = torch.ones(size=[args.batch_size], device='cuda').bool().detach()
    directly_update_contact_points_flag = torch.zeros(size=[args.batch_size], device='cuda').detach()
    new_z = z_x.clone()

  update_indices = torch.randint(0, 3, size=[args.batch_size], device='cuda')
  # choose next point smarter
  to_src_dist = hand_model.mano_manifold_distances[contact_point_indices_x]
  to_tgt_dist = hand_model.mano_manifold_distances[contact_point_indices_star]
  prob = 1 / (to_src_dist + to_tgt_dist.sum(1, keepdim=True))  # B x 3 x V
  prob = prob[torch.arange(args.batch_size), update_indices]  # B x V
  update_to = torch.cat([torch.multinomial(prob[b], 1) for b in range(args.batch_size)])
  # update_to = torch.randint(0, hand_model.num_points, size=[args.batch_size], device='cuda')
  temp_new_contact_point_indices = contact_point_indices_x.clone()
  temp_new_contact_point_indices[torch.arange(args.batch_size), update_indices] = update_to
  new_contact_point_indices[directly_update_contact_points] = temp_new_contact_point_indices[directly_update_contact_points]
  new_z[directly_update_contact_points] = z_x[directly_update_contact_points]
  directly_update_contact_points_flag[directly_update_contact_points] = 0 # not accepted changes may not be 0

  # compute new energy
  new_energy = compute_energy(object_x, new_z, new_contact_point_indices)
  with torch.no_grad():
    new_d =  distance([object_x, new_z, new_contact_point_indices], [object_star, z_star, contact_point_indices_star])
    # metropolis-hasting
    _alpha = torch.rand(z_x.shape[0], device='cuda').float()
    accept = _alpha < torch.exp((energy - new_energy + alpha * (d - new_d)) / T)
    z_x[accept] = new_z[accept]
    d[accept] = new_d[accept]
    contact_point_indices_x[accept] = new_contact_point_indices[accept]
    energy[accept] = new_energy[accept]
  return [object_x, z_x.detach(), contact_point_indices_x.detach()], d.detach(), energy.detach(), directly_update_contact_points_flag.detach()

def attraction_diffusion(C, Xstar, alpha, delta, T, M):
  directly_update_contact_points_flag = torch.zeros(size=[args.batch_size], device='cuda')
  d = distance(C, Xstar)
  dstar = d
  m = torch.zeros([d.shape[0]], device='cuda').float()
  B = None
  finished = ((d <= delta).sum() > 0) or ((m < M).sum() == 0)
  not_finished = torch.logical_and(d > delta, m < M)
  states = [[x.detach().cpu().numpy() for x in C]]
  while not finished:
    C, d, U, directly_update_contact_points_flag = MCMC(C, Xstar, T, alpha, directly_update_contact_points_flag.detach())
    # update energy barrier
    if B is None:
      B = U.clone()
    else:
      B_update = torch.logical_and(not_finished, B < U)
      B[B_update] = U[B_update].clone()
    d_update_1 = torch.logical_and(not_finished, d >= dstar)
    d_update_2 = torch.logical_and(not_finished, d < dstar)
    m[d_update_1] = m[d_update_1] + 1
    m[d_update_2] = 0
    dstar[d_update_2] = d[d_update_2]
    finished = ((d <= delta).sum() > 0) or ((m < M).sum() == 0)
    not_finished = torch.logical_and(d > delta, m < M)
    states.append([x.detach().cpu().numpy() for x in C])
  if B is None:
    B = torch.zeros([d.shape[0]], device='cuda').float()
  states.append([x.detach().cpu().numpy() for x in Xstar])
  return d, B, states

def load_proposals(path):
  if os.path.exists('data/proposals.pkl'):
    Y, energies = pickle.load(open('data/proposals.pkl', 'rb'))
  else:
    Y = []
    energies = []
    for fn in os.listdir(path):
      obj_code, z, contact_point_indices, energy, _, _, _ = pickle.load(open(os.path.join(path, fn), 'rb'))
      energy = energy.detach().cpu().numpy()
      for i in range(len(obj_code)):
        linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code[[i]], z[[i]], contact_point_indices[[i]], verbose=True)
        total_energy = (linear_independence + force_closure + surface_distance + penetration + z_norm + normal_alignment).squeeze().detach().cpu().numpy()
        if force_closure.squeeze().data < 0.01 and surface_distance.squeeze().data < 0.02 and penetration.squeeze().data < 0.02 and z_norm.squeeze().data < 5 and total_energy < 1:
          Y.append((obj_code[i], z[i], contact_point_indices[i]))
          energies.append(total_energy)
    energies = np.array(energies)
    pickle.dump([Y, energies], open('data/proposals.pkl', 'wb'))
  return Y[:10], energies[:10]

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

def clone(X):
  return [x.clone() for x in X]

def draw(Y, l, i):
  obj, z, idx = Y
  mesh = get_obj_mesh_by_code(obj)
  hand_v = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  fig = go.Figure(data=[
    go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2], i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2], color='lightblue'),
    go.Mesh3d(x=hand_v[:,0], y=hand_v[:,1], z=hand_v[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink')
  ])
  fig.update_layout(
    dict1=dict(autosize=True, margin={'l':0, 'r': 0, 't': 0, 'b': 0}), 
    scene=dict(
      xaxis=dict(showticklabels=False, title_text=''), 
      yaxis=dict(showticklabels=False, title_text=''), 
      zaxis=dict(showticklabels=False, title_text=''), 
      ))
  if not args.time:
    os.makedirs('adelm_result/%d'%l, exist_ok=True)
    fig.write_image('adelm_result/%d/%d.png'%(l, i))

examples, example_energies = load_proposals(args.data_path)  # each proposal is already a local minimum

t00 = time.time()
# ADELM
basin_labels = []
basin_minima = []
basin_minima_energies = []
def inf():
  return float('inf')

energy_barriers: DefaultDict[Tuple[int, int], float] = defaultdict(inf)
mc_chains: Dict[Tuple[int, int], List] = {}

for n in range(len(examples)):
  t0 = time.time()
  barrier_to_basins = []
  minima_updated = False
  if n == 0:
    basin_minima.append(examples[n])
    basin_minima_energies.append(example_energies[n])
    minima_updated = True
    basin_labels.append(0)
    number_of_basins = 1
  else:
    number_of_basins = max(basin_labels) + 1
    candidate_labels = []

    C = [[] for _ in range(len(basin_minima))]
    barrier_to_basins = [float('inf') for _ in range(len(basin_minima))]

    for j in range(len(basin_minima)):
      Yn = tile(examples[n], args.batch_size)
      Zs = tile(basin_minima[j], args.batch_size)

      d1, B1, C1 = attraction_diffusion(clone(Yn), clone(Zs), args.alpha, args.delta, args.T, args.M)
      d2, B2, C2 = attraction_diffusion(clone(Zs), clone(Yn), args.alpha, args.delta, args.T, args.M)

      B1 = B1[d1 < args.delta].detach().cpu().numpy()
      B2 = B2[d2 < args.delta].detach().cpu().numpy()

      d1 = d1.detach().cpu().numpy()
      d2 = d2.detach().cpu().numpy()

      i1 = np.argmin(d1)
      i2 = np.argmin(d2)

      d1min = d1[i1]
      d2min = d2[i2]

      print('item %d to basin %d: distances = %f, %f'%(n, j, d1min, d2min))

      if min(d1min, d2min) < args.delta:
        candidate_labels.append(j)
        if d1min > args.delta:
          c2i1 = d2 < args.delta
          c2i2 = np.argmin(B2)
          C2 = [(x[c2i1][c2i2], y[c2i1][c2i2], z[c2i1][c2i2]) for x, y, z in C2]
          barrier_to_basins[j] = B2.min()
          mc_chains[(n,j)] = C2
        elif d2min > args.delta:
          c1i1 = d1 < args.delta
          c1i2 = np.argmin(B1)
          C1 = [(x[c1i1][c1i2], y[c1i1][c1i2], z[c1i1][c1i2]) for x, y, z in C1]
          barrier_to_basins[j] = B1.min()
          mc_chains[(n,j)] = C1
        else:
          if B1.min() < B2.min():
            c1i1 = d1 < args.delta
            c1i2 = np.argmin(B1)
            C1 = [(x[c1i1][c1i2], y[c1i1][c1i2], z[c1i1][c1i2]) for x, y, z in C1]
            barrier_to_basins[j] = B1.min()
            mc_chains[(n,j)] = C1
          else:
            c2i1 = d2 < args.delta
            c2i2 = np.argmin(B2)
            C2 = [(x[c2i1][c2i2], y[c2i1][c2i2], z[c2i1][c2i2]) for x, y, z in C2]
            barrier_to_basins[j] = B2.min()
            mc_chains[(n,j)] = C2

    if len(candidate_labels) == 0:
      basin_minima.append(examples[n])
      basin_minima_energies.append(example_energies[n])
      basin_labels.append(number_of_basins)
      minima_updated = True
    else:
      basin_labels.append(np.argmin(barrier_to_basins))
      if example_energies[n] < basin_minima_energies[basin_labels[n]]:
        basin_minima_energies[basin_labels[n]] = example_energies[n]
        basin_minima[basin_labels[n]] = examples[n]
        minima_updated = True
    
    current_label = basin_labels[-1]
    for j in range(number_of_basins):
      if energy_barriers[(current_label, j)] > barrier_to_basins[j]:
        energy_barriers[(current_label, j)] = barrier_to_basins[j]
        energy_barriers[(j, current_label)] = barrier_to_basins[j]
        
      print('barrier(%d,%d): %f'%(current_label, j, barrier_to_basins[j]))

  print('%d-th data falls in basin #%d (total: %d). Time: %f'%(n, basin_labels[n], max(basin_labels) + 1, time.time() - t0))
  if not args.time:
    draw(examples[n], basin_labels[n], n)
    pickle.dump([basin_labels[n], barrier_to_basins], open('adelm_result/%d/%d.pkl'%(basin_labels[n], n), 'wb'))
    if minima_updated:
      pickle.dump([basin_minima[basin_labels[n]], basin_minima_energies[basin_labels[n]]], open('adelm_result/%d/minima.pkl'%(basin_labels[n]), 'wb'))

if not args.time:
  pickle.dump([basin_labels, basin_minima, basin_minima_energies, energy_barriers, mc_chains], open('ADELM.pkl', 'wb'))

t1 = time.time()
print('Total time: ', t1-t00)
