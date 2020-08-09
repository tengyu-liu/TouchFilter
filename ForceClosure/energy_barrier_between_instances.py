"""
ADELM requires sampling monte carlo chains between states and local minima that minimizes both energy and distance to target. 
In the grasping case, it is non-trivial to define a distance metric between graspings of different objects that reflects the
taxonomy of grasping. So instead of finding the minimum barrier MC chain, we directly estimate the minimum pairwise energy 
barrier between states. 
"""

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
parser.add_argument('--data_path', default='logs/sample', type=str)
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

def load_proposals(path):
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
  print(len(Y))
  return Y, np.array(energies)

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

examples, example_energies = load_proposals(args.data_path)

objs, zs, indices = collate(examples)

barrier = np.zeros([len(examples), len(examples)])

batchsize = 65
n_batch = len(example_energies) // batchsize

for i in range(len(examples)):
  print('\r', i, len(examples), end='')
  for batch_id in range(n_batch+1):
    obj_i, z_i, idx_i = tile(examples[i], indices[batch_id * batchsize : (batch_id+1) * batchsize].shape[0])
    d1 = compute_energy(obj_i, z_i, indices[batch_id * batchsize : (batch_id+1) * batchsize]).detach().cpu().numpy()
    d2 = compute_energy(objs[batch_id * batchsize : (batch_id+1) * batchsize], zs[batch_id * batchsize : (batch_id+1) * batchsize], idx_i).detach().cpu().numpy()
    d = np.minimum(d1, d2)
    e = np.maximum(example_energies[i], example_energies[batch_id * batchsize : (batch_id+1) * batchsize])
    d = np.maximum(d, e)
    barrier[i, batch_id * batchsize : (batch_id+1) * batchsize] = d
    barrier[batch_id * batchsize : (batch_id+1) * batchsize, i] = d

np.save('barrier.npy', barrier)

import matplotlib.pyplot as plt
plt.imshow(barrier)
plt.colorbar()
plt.show()
