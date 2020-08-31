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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--n_contact', default=3, type=int)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--name', default='exp', type=str)
parser.add_argument('--id', default=0, type=int)
args = parser.parse_args()

np.seterr(all='raise')
random.seed(args.id)
np.random.seed(args.id)
torch.manual_seed(args.id)
args.name = args.name + '_%d'%args.id

from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

hand_model = HandModel(flat_hand_mean=False)
object_model = ObjectModel()
fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

log_dir = os.path.join('logs', args.name)
obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history = pickle.load(open(os.path.join(log_dir, 'saved_998000.pkl'), 'rb'))

def compute_energy(obj_code, z, contact_point_indices, verbose=False, sd_weight=1):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(args.batch_size), contact_point_indices[:,i],:] for i in range(args.n_contact)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(z=z)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(args.n_contact)], dim=1)
  hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
  normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
  surface_distance = fc_loss_model.dist_loss(obj_code, contact_point) * sd_weight
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-6:], dim=-1) * 0.1
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment

mask = torch.tensor(np.eye(15)).float().cuda().unsqueeze(0)  # 1 x 6 x 6

energy = compute_energy(obj_code, z, contact_point_indices, sd_weight=100)
old_energy = energy.clone()
grad = torch.autograd.grad(energy.sum(), z)[0]
grad_ema = EMA(args.mu)
grad_ema.apply(grad)

for _iter in range(10000):
    step_size = 0.1
    temperature = 1e-3
    noise = torch.normal(mean=0, std=1, size=z.shape, device='cuda').float() * step_size
    new_z = z - 0.5 * grad * mask[:,_iter%15,:] / grad_ema.average.unsqueeze(0) * step_size * step_size + noise
    new_energy = compute_energy(obj_code, new_z, contact_point_indices, sd_weight=100)
    new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
    alpha = torch.rand(args.batch_size, device='cuda').float()
    accept = (alpha < torch.exp((energy - new_energy) / temperature)).long()
    z = z * (1-accept.unsqueeze(1)) + new_z * accept.unsqueeze(1)
    energy = energy * (1-accept) + new_energy * accept
    grad = grad * (1-accept.unsqueeze(1)) + new_grad * accept.unsqueeze(1)
    grad_ema.apply(grad)
    if _iter % 100 == 0:
        print(_iter, (energy-old_energy).mean().detach().cpu().numpy(), accept.float().mean().detach().cpu().numpy())

pickle.dump([obj_code, z, contact_point_indices], open(os.path.join(log_dir, 'optim.pkl'), 'wb'))
