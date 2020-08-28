import sys
import pickle

import numpy as np
import torch
import plotly
import plotly.graph_objects as go

from CodeUtil import *
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel
from EMA import EMA

batch_size = 256
step_size = 0.1
annealing_period = 3000
starting_temperature = 0.1
temperature_decay = 0.95
stepsize_period = 3000
noise_size = 0.1

hand_model = HandModel(
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False,
  n_contact=5)
object_model = ObjectModel()
fc_loss = FCLoss(object_model)
penetration_model = PenetrationModel(hand_model, object_model)

def compute_energy(obj_code, z, contact_point_indices, verbose=False, sd_weight=1):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(batch_size), contact_point_indices[:,i],:] for i in range(5)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(z=z)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(5)], dim=1)
  hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
  normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure = fc_loss.fc_loss(contact_point, contact_normal, obj_code)
  surface_distance = fc_loss.dist_loss(obj_code, contact_point) * sd_weight
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-6:], dim=-1) * 0.1
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment

data = []
fltr = []
for i in range(batch_size):
  linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code[[i]], z[[i]], contact_point_indices[[i]], sd_weight=1, verbose=True)
  if force_closure.sum() < 0.5 and surface_distance.sum() < 0.01 and penetration < 0.01:
    fltr.append(i)
