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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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
parser.add_argument('--n_contact', default=3, type=int)
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
parser.add_argument('--data_path', default='logs/mcmc', type=str)
parser.add_argument('--log_path', default='data', type=str)
parser.add_argument('--time', action='store_true')
args = parser.parse_args()

os.makedirs(args.log_path, exist_ok=True)
os.makedirs(os.path.join(args.log_path, 'adelm_result'), exist_ok=True)
os.makedirs(os.path.join(args.log_path, 'item_basin_barriers'), exist_ok=True)

# prepare models
hand_model = HandModel(
  n_handcode=args.n_handcode,
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False,
  mano_path=args.mano_path,
  n_contact=args.n_contact)

object_model = ObjectModel(
  state_dict_path="data/ModelParameters/2000.pth"
)

grad_ema = EMA(args.mu)

fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# compute energy
def compute_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(z.shape[0]), contact_point_indices[:,i],:] for i in range(args.n_contact)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(verts=hand_verts)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(args.n_contact)], dim=1)
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
  distance = hand_model.manifold_distance(contact_point_indices_x, contact_point_indices_y)
  return distance

def MCMC(X, Xstar, T, alpha, directly_update_contact_points_flag):
  object_x, z_x, contact_point_indices_x = X
  object_x.requires_grad_()
  z_x.requires_grad_()
  object_star, z_star, contact_point_indices_star = Xstar
  batch_size = len(z_x)
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
    directly_update_contact_points = torch.ones(size=[batch_size], device='cuda').bool().detach()
    directly_update_contact_points_flag = torch.zeros(size=[batch_size], device='cuda').detach()
    new_z = z_x.clone()
  update_indices = torch.randint(0, args.n_contact, size=[batch_size], device='cuda')
  # choose next point smarter
  to_src_dist = hand_model.mano_manifold_distances[contact_point_indices_x]
  to_tgt_dist = hand_model.mano_manifold_distances[contact_point_indices_star]
  prob = 1 / (to_src_dist + to_tgt_dist.sum(1, keepdim=True))  # B x n_contact x V
  prob = prob[torch.arange(batch_size), update_indices]  # B x V
  update_to = torch.cat([torch.multinomial(prob[b], 1) for b in range(batch_size)])
  # update_to = torch.randint(0, hand_model.num_points, size=[batch_size], device='cuda')
  temp_new_contact_point_indices = contact_point_indices_x.clone()
  temp_new_contact_point_indices[torch.arange(batch_size), update_indices] = update_to
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
    directly_update_contact_points_flag[~accept] += 1
  return [object_x.detach(), z_x.detach(), contact_point_indices_x.detach()], d.detach(), energy.detach(), directly_update_contact_points_flag.detach()

def load_proposals(path):
  if os.path.exists(os.path.join(args.log_path, 'proposals.pkl')):
    Y, energies = pickle.load(open(os.path.join(args.log_path, 'proposals.pkl'), 'rb'))
  else:
    Y = []
    energies = []
    li = []
    fc = []
    sd = []
    pen = []
    zn = []
    na = []
    contact_point_indices_all = []
    for fn in os.listdir(path):
      obj_code, z, contact_point_indices, energy, _, _, _ = pickle.load(open(os.path.join(path, fn), 'rb'))
      contact_point_indices_all.append(contact_point_indices)
      energy = energy.detach().cpu().numpy()
      for i in range(len(obj_code)):
        linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code[[i]], z[[i]], contact_point_indices[[i]], verbose=True)
        li.append(linear_independence.squeeze().item())
        fc.append(force_closure.squeeze().item())
        sd.append(surface_distance.squeeze().item())
        pen.append(penetration.squeeze().item())
        zn.append(z_norm.squeeze().item())
        na.append(normal_alignment.squeeze().item())
        total_energy = (linear_independence + force_closure + surface_distance + penetration + z_norm + normal_alignment).squeeze().detach().cpu().numpy()
        if force_closure.squeeze().data < 0.5 and surface_distance.squeeze().data < 1e-2 and penetration.squeeze().data < 1e-2 and z_norm.squeeze().data < 3.5:
          Y.append((obj_code[i], z[i], contact_point_indices[i]))
          energies.append(total_energy)
    energies = np.array(energies)
    pickle.dump([Y, energies], open(os.path.join(args.log_path, 'proposals.pkl'), 'wb'))
  return Y, energies

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
  os.makedirs(os.path.join(args.log_path, 'adelm_result/%d'%l), exist_ok=True)
  fig.write_html(os.path.join(args.log_path, 'adelm_result/%d/%d.html'%(l, i)))

def combine(Y, x):
  return torch.cat([Y[0], x[0].unsqueeze(0)], dim=0), torch.cat([Y[1], x[1].unsqueeze(0)], dim=0), torch.cat([Y[2], x[2].unsqueeze(0)], dim=0)

examples, example_energies = load_proposals(args.data_path)  # each proposal is already a local minimum

t0 = time.time()

# ADELM
basin_labels = [-1 for i in range(len(examples))]
basin_labels[0] = 0
num_basins = 1
item_basin_barriers = [[float('inf')] for i in range(len(examples))]  # keep track of energy barriers from items to basins
item_basin_barriers[0][0] = example_energies[0]
basin_minima = [examples[0]]
basin_minima_energies = [example_energies[0]]
job_queue:List[Tuple[int,int,int]] = []
current_job:List[Tuple[int,int,int]] = []
Y = []
Z = []
YE = []
ZE = []

draw(examples[0], 0, 0)
shutil.copy(os.path.join(args.log_path, 'adelm_result', '0', '0.html'), os.path.join(args.log_path, 'adelm_result', '0', 'minima.html'))

job_queue_item_count = np.zeros([len(examples)])
for i in range(1, len(examples)):
  job_queue.append((i,0,0))
  job_queue.append((i,0,1))
  job_queue_item_count[i] += 2

for _ in range(min(args.batch_size, len(job_queue))):
  current_job.append(job_queue.pop(0))
  item, basin, direction = current_job[-1]
  if direction == 0:
    Y.append(examples[item])
    Z.append(basin_minima[basin])
    YE.append(example_energies[item])
    ZE.append(basin_minima_energies[basin])
  else:
    Y.append(basin_minima[basin])
    Z.append(examples[item])
    YE.append(basin_minima_energies[basin])
    ZE.append(example_energies[item])

Y = collate(Y)
Z = collate(Z)
YET = torch.tensor(YE).cuda()
ZET = torch.tensor(ZE).cuda()

def inf():
  return float('inf')

# initialize variables for attraction-diffusion
flag = torch.zeros(size=[len(YET)], device='cuda')
d = distance(Y, Z).detach()
dstar = d.clone()
m = torch.zeros([len(d)], device='cuda').float()
B = torch.max(YET, ZET)

success_step = 0

while (len(job_queue) + len(current_job)) > 0:
  # removal
  success = d <= args.delta
  failure = m > args.M
  if success.sum() > 0:
    for i in range(len(success)):
      if success[i].item():
        # if success
        item_id, basin_label, direction = current_job[i]
        job_queue_item_count[item_id] -= 1
        # update basin label
        if B[i].item() < min(item_basin_barriers[item_id]):
          if basin_labels[item_id] > -1:
            if not args.time:
              os.remove(os.path.join(args.log_path, 'adelm_result', str(basin_labels[item_id]), '%d.html'%item_id))
          basin_labels[item_id] = basin_label
          if not args.time:
            draw(examples[item_id], basin_label, item_id)
          print('successful AD from item #%d to basin #%d with barrier %f'%(item_id, basin_label, B[i].item()))
          # if is new basin minima: 
          if example_energies[item_id] < basin_minima_energies[basin_label]:
            basin_minima_energies[basin_label] = example_energies[item_id]
            basin_minima[basin_label] = examples[item_id]
            print('    item #%d is the new basin minima'%item_id)
            if not args.time:
              shutil.copy(os.path.join(args.log_path, 'adelm_result', str(basin_label), '%d.html'%item_id), os.path.join(args.log_path, 'adelm_result', str(basin_label), 'minima.html'))
        # update basin barrier
        item_basin_barriers[item_id][basin_label] = min(item_basin_barriers[item_id][basin_label], B[i].item())
    # save basin barrier
    if not args.time:
      pickle.dump(item_basin_barriers, open(os.path.join(args.log_path, 'item_basin_barriers/%d.pkl'%success_step), 'wb'))
    success_step += 1
  if failure.sum() > 0:
    for i in range(len(failure)):
      if failure[i].item():
        # if failure
        item_id, basin_label, direction = current_job[i]
        job_queue_item_count[item_id] -= 1
        print('rejecting AD from item #%d to basin #%d'%(item_id, basin_label))
        # check to create new minima
        if job_queue_item_count[item_id] == 0 and all([x == float('inf') for x in item_basin_barriers[item_id]]):
          basin_labels[item_id] = num_basins
          basin_minima.append(examples[item_id])
          basin_minima_energies.append(example_energies[item_id])
          for _item_id in range(len(examples)):
            if _item_id != item_id:
              job_queue.append((_item_id, num_basins, 0))
              job_queue.append((_item_id, num_basins, 1))
          job_queue_item_count += 2
          job_queue_item_count[item_id] -= 2
          for _item_id in range(len(examples)):
            if _item_id == item_id:
              item_basin_barriers[_item_id].append(example_energies[item_id])
            else:             
              item_basin_barriers[_item_id].append(float('inf'))
          print('    assign new basin #%d for item #%d'%(num_basins, item_id))
          if not args.time:
            os.makedirs(os.path.join(args.log_path, 'adelm_result', str(num_basins)), exist_ok=True)
            draw(examples[item_id], num_basins, item_id)
            shutil.copy(os.path.join(args.log_path, 'adelm_result', str(num_basins), '%d.html'%item_id), os.path.join(args.log_path, 'adelm_result', str(num_basins), 'minima.html'))
          num_basins += 1
  if success.sum() + failure.sum() > 0:
    for i in range(len(success)-1,-1,-1):
      if success[i].item() or failure[i].item():
        tmp = current_job.pop(i)
    fltr = ~(success+failure)
    d = d[fltr]
    m = m[fltr]
    B = B[fltr]
    dstar = dstar[fltr]
    flag = flag[fltr]
    Y = (Y[0][fltr], Y[1][fltr], Y[2][fltr])
    Z = (Z[0][fltr], Z[1][fltr], Z[2][fltr])
    # refill
    refill_count = min(args.batch_size - len(current_job), len(job_queue))
    YE = []
    ZE = []
    for i in range(refill_count):
      current_job.append(job_queue.pop(0))
      item, basin, direction = current_job[-1]
      if direction == 0:
        Y = combine(Y, examples[item])
        Z = combine(Z, basin_minima[basin])
        YE.append(example_energies[item])
        ZE.append(basin_minima_energies[basin])
      else:
        Y = combine(Y, basin_minima[basin])
        Z = combine(Z, examples[item])
        YE.append(basin_minima_energies[basin])
        ZE.append(example_energies[item])
    if refill_count > 0:
      m = torch.cat([m, torch.zeros([refill_count], device='cuda')], dim=0)
      d = torch.cat([d, distance([y[-refill_count:] for y in Y], [z[-refill_count:] for z in Z])], dim=0)
      dstar = torch.cat([dstar, d[-refill_count:]], dim=0)
      flag = torch.cat([flag, torch.zeros([refill_count], device='cuda')])
      YET = torch.tensor(YE, device='cuda')
      ZET = torch.tensor(ZE, device='cuda')
      B = torch.cat([B, torch.max(YET, ZET)], dim=0)
  if len(current_job) == 0:
    continue
  # run attraction-diffusion between Y and Z
  assert len(current_job) == len(Y[0])
  assert len(current_job) == len(list(set(current_job)))
  print('\rrunning MCMC on %d/%d jobs    '%(len(Y[0]), len(job_queue)), end='')
  Y, d, U, flag = MCMC(Y, Z, args.T, args.alpha, flag)
  B[U>B] = U[U>B].clone()
  m[d>=dstar] = m[d>=dstar] + 1
  m[d<dstar] = 0
  dstar[d<dstar] = d[d<dstar]

if not args.time:
  pickle.dump([basin_labels, basin_minima, basin_minima_energies, item_basin_barriers], open(os.path.join(args.log_path, 'ADELM_dispatch.pkl'), 'wb'))

t1 = time.time()
print('Total time:', t1 - t0)