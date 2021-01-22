"""
TODO: 
0. read data
1. optimize result for 3000 steps
2. visualize reesult
"""

import random
import sys
import pickle
import os

import plotly
import plotly.graph_objects as go

# from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

import torch
from torch import nn
import numpy as np
import trimesh as tm

fn = sys.argv[1]

hand_model = HandModel(flat_hand_mean=False)
object_model = ObjectModel(
  state_dict_path='F:\\OverfitSDF\\weights\\2049999.pth'
)
fc_loss = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

data = pickle.load(open(fn, 'rb'))

obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history, z_history, contact_point_history, individual_energy_history, cats_and_names = data
batch_size = z.shape[0]

# i_item = np.argmin(energy.detach().cpu().numpy())
# i_item = int(random.random() * len(z))
# i_item = 53
# print(i_item)

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, tm.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = tm.util.concatenate(
                tuple(tm.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, tm.Trimesh))
        mesh = scene_or_mesh
    return mesh

def compute_energy(obj_code, z, contact_point_indices, verbose=False, sd_weight=1):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(batch_size), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
  hand_normal = hand_model.get_surface_normals(z=z)
  hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(3)], dim=1)
  hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
  normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure = fc_loss.fc_loss(contact_point, contact_normal, obj_code)
  surface_distance = fc_loss.dist_loss(obj_code, contact_point)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-6:], dim=-1) * 0.1
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment

def draw_with_obj(o,z,i,f):
  verts = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  obj_mesh = as_mesh(tm.load(os.path.join('F:\\dataset\\ShapeNetCore.v2', o[0], o[1], 'models/model_normalized.obj')))
  i = i.detach().cpu().numpy()
  faces = hand_model.faces
  fig = go.Figure(data=[
    go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink'),
    go.Scatter3d(x=verts[i,0], y=verts[i,1], z=verts[i,2], mode='markers', marker=dict(size=5, color='red')),
    go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue')
  ])
  # fig.write_html(f + '.html')
  # fig.write_image(f + '.png')
  fig.show()
  input()

old_z = z.clone()

grad_ema = EMA(0.98)

energy = compute_energy(obj_code, z, contact_point_indices, sd_weight=100)
grad = torch.autograd.grad(energy.sum(), z)[0]
mean_energy = []

import matplotlib.pyplot as plt
plt.ion()
mask = torch.tensor(np.eye(15)).float().cuda().unsqueeze(0)  # 1 x 6 x 6

# for _iter in range(10000):
#   step_size = 0.1
#   temperature = 1e-3
#   noise = torch.normal(mean=0, std=1, size=z.shape, device='cuda').float() * step_size
#   new_z = z - 0.5 * grad * mask[:,_iter%15,:] / grad_ema.average.unsqueeze(0) * step_size * step_size + noise
#   new_energy = compute_energy(obj_code, new_z, contact_point_indices, sd_weight=100)
#   new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
#   alpha = torch.rand(batch_size, device='cuda').float()
#   accept = (alpha < torch.exp((energy - new_energy) / temperature)).long()
#   z = z * (1-accept.unsqueeze(1)) + new_z * accept.unsqueeze(1)
#   energy = energy * (1-accept) + new_energy * accept
#   grad = grad * (1-accept.unsqueeze(1)) + new_grad * accept.unsqueeze(1)
#   grad_ema.apply(grad)
#   mean_energy.append(energy.mean().detach().cpu().numpy())
#   if _iter % 10 == 0:
#     _ = plt.clf()
#     _ = plt.plot(mean_energy)
#     _ = plt.pause(1e-5)

linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code, z, contact_point_indices, sd_weight=1, verbose=True)
# plt.savefig('aaai/figs/functional/%d.png'%j)
for i in range(batch_size):
  if force_closure[i] > 0.5 or surface_distance[i] > 0.02 or penetration[i] > 0.02:
      continue
  print(i)
  print('\tlinear_independence', linear_independence[i])
  print('\tforce_closure', force_closure[i])
  print('\tsurface_distance', surface_distance[i])
  print('\tpenetration', penetration[i])
  print('\tz_norm', z_norm[i])
  print('\tnormal_alignment', normal_alignment[i])
  draw_with_obj(cats_and_names[obj_code[i]], old_z[i], contact_point_indices[i], '')
  # draw_with_obj(cats_and_names[i], z[i], contact_point_indices[i], '')
#   pickle.dump([obj_code[i], z[i], contact_point_indices[i]], open('aaai/figs/functional/%d/syn2_%d.pkl'%(i,j), 'wb'))
