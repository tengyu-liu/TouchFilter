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
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

hand_model = HandModel(flat_hand_mean=False, n_contact=5)
object_model = ObjectModel()
fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

def compute_energy(obj_code, z, contact_point_indices, verbose=False, sd_weight=1):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(len(z)), contact_point_indices[:,i],:] for i in range(5)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(z=z)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(5)], dim=1)
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

ocs = []
zs = []
cpis = []

li, fc, sd, pen, zn, na = [],[],[],[],[],[]

for fn in os.listdir('logs/zeyu_5p'):
    if 'optim' in fn:
        obj_code, z, contact_point_indices = pickle.load(open(os.path.join('logs/zeyu_5p', fn), 'rb'))
        ocs.append(obj_code)
        zs.append(z)
        cpis.append(contact_point_indices)
        for i in range(len(z)):
            linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code[[i]], z[[i]], contact_point_indices[[i]], verbose=True)
            li.append(linear_independence.detach().cpu().numpy())
            fc.append(force_closure.detach().cpu().numpy())
            sd.append(surface_distance.detach().cpu().numpy())
            pen.append(penetration.detach().cpu().numpy())
            zn.append(z_norm.detach().cpu().numpy())
            na.append(normal_alignment.detach().cpu().numpy())

li, fc, sd, pen, zn, na = list(map(np.array, [li, fc, sd, pen, zn, na]))

ocs = torch.cat(ocs, dim=0)
zs = torch.cat(zs, dim=0)
cpis = torch.cat(cpis, dim=0)

idx = (fc < 0.5) * (sd < 0.01) * (pen<0.01) * (zn < 3.5)
idx = idx[:,0]
ocs_sel = ocs[idx]
zs_sel = zs[idx]
cpis_sel = cpis[idx]
pickle.dump([ocs_sel, zs_sel, cpis_sel, li[idx,0], fc[idx,0], sd[idx,0], pen[idx,0], zn[idx,0], na[idx,0]], open('aaai/supplementary/code_and_data/results/ADELM_candidate.pkl', 'wb'))
