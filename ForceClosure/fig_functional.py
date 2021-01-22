import pickle

import numpy as np
import torch
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
_ = plt.ion()

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

os.makedirs('aaai/figs/functional', exist_ok=True)
for i in range(batch_size):
  os.makedirs('aaai/figs/functional/%d'%i, exist_ok=True)

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
  surface_distance = fc_loss.dist_loss(obj_code, contact_point)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-6:], dim=-1) * 0.1
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment

def T(t):
  return starting_temperature * temperature_decay ** (t // annealing_period)

def S(t):
  return noise_size * temperature_decay ** (t // stepsize_period)

def draw_no_obj(z, i, f):
  verts = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  i = i.detach().cpu().numpy()
  faces = hand_model.faces
  fig = go.Figure(data=[
    go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink'),
    go.Scatter3d(x=verts[i,0], y=verts[i,1], z=verts[i,2], mode='markers', marker=dict(size=20, color='red'))
  ])
  fig.write_html(f + '.html')
  fig.write_image(f + '.png')

def draw_with_obj(o,z,i,f):
  verts = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  obj_mesh = get_obj_mesh_by_code(o)
  i = i.detach().cpu().numpy()
  faces = hand_model.faces
  fig = go.Figure(data=[
    go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=faces[:,0], j=faces[:,1], k=faces[:,2], color='lightpink'),
    # go.Scatter3d(x=verts[i,0], y=verts[i,1], z=verts[i,2], mode='markers', marker=dict(size=5, color='red')),
    go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue')
  ])
  fig.write_html(f + '.html')
  fig.write_image(f + '.png')

# 1. load an object and a grasp point
contact_point_indices = []
adelm_base_path = 'adelm_7/ADELM_dispatch.pkl'
basin_labels, basin_minima, basin_minima_energies, item_basin_barriers = pickle.load(open(adelm_base_path, 'rb'))

for i, idx in enumerate(np.argsort(basin_minima_energies)):
  print(i)
  _, _, cpi = basin_minima[idx]
  contact_point_indices = cpi
  _z = torch.normal(0,1,size=[hand_model.code_length]).float().cuda() * 1e-6
  draw_no_obj(_z, cpi, 'aaai/figs/functional/%d/query'%i)

  contact_point_indices = torch.unsqueeze(contact_point_indices, dim=0).repeat(batch_size, 1)

  obj_code, obj_idx = get_obj_code_random(batch_size, 256)
  z = torch.normal(mean=0, std=1, size=[batch_size, hand_model.code_length], requires_grad=True).float().cuda() * 1e-5
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
  _ = z.requires_grad_()
  # 2. minimize loss
  energy = compute_energy(obj_code, z, contact_point_indices)
  grad = torch.autograd.grad(energy.sum(), z)[0]
  grad_ema = EMA(0.98)
  grad_ema.apply(grad)
  mean_energy = []
  for _iter in range(10000):
    step_size = S(_iter)
    temperature = T(_iter)
    noise = torch.normal(mean=0, std=1, size=z.shape, device='cuda').float() * step_size
    new_z = z - 0.5 * grad / grad_ema.average.unsqueeze(0) * step_size * step_size + noise
    new_energy = compute_energy(obj_code, new_z, contact_point_indices)
    new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
    alpha = torch.rand(batch_size, device='cuda').float()
    accept = (alpha < torch.exp((energy - new_energy) / temperature)).long()
    z = z * (1-accept.unsqueeze(1)) + new_z * accept.unsqueeze(1)
    energy = energy * (1-accept) + new_energy * accept
    grad = grad * (1-accept.unsqueeze(1)) + new_grad * accept.unsqueeze(1)
    grad_ema.apply(new_grad)
    mean_energy.append(energy.mean().detach().cpu().numpy())
    if _iter % 10 == 0:
      _ = plt.clf()
      _ = plt.plot(mean_energy)
      _ = plt.yscale('log')
      _ = plt.pause(1e-5)

  mask = torch.tensor(np.eye(15)).float().cuda().unsqueeze(0)  # 1 x 6 x 6

  energy = compute_energy(obj_code, z, contact_point_indices, sd_weight=100)
  grad = torch.autograd.grad(energy.sum(), z)[0]
  for _iter in range(10000):
    step_size = 0.1
    temperature = 1e-3
    noise = torch.normal(mean=0, std=1, size=z.shape, device='cuda').float() * step_size
    new_z = z - 0.5 * grad * mask[:,_iter%15,:] / grad_ema.average.unsqueeze(0) * step_size * step_size + noise
    new_energy = compute_energy(obj_code, new_z, contact_point_indices, sd_weight=100)
    new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
    alpha = torch.rand(batch_size, device='cuda').float()
    accept = (alpha < torch.exp((energy - new_energy) / temperature)).long()
    z = z * (1-accept.unsqueeze(1)) + new_z * accept.unsqueeze(1)
    energy = energy * (1-accept) + new_energy * accept
    grad = grad * (1-accept.unsqueeze(1)) + new_grad * accept.unsqueeze(1)
    grad_ema.apply(new_grad)
    mean_energy.append(energy.mean().detach().cpu().numpy())
    if _iter % 10 == 0:
      _ = plt.clf()
      _ = plt.plot(mean_energy)
      _ = plt.yscale('log')
      _ = plt.pause(1e-5)
  linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code, z, contact_point_indices, verbose=True)
  for j in range(batch_size):
    if force_closure[j] < 0.5 and surface_distance[j] < 1e-2 and penetration[j] < 1e-2:
      draw_with_obj(obj_code[j], z[j], contact_point_indices[j], 'aaai/figs/functional/%d/syn_%d'%(i,j))
      pickle.dump([obj_code[j], z[j], contact_point_indices[j]], open('aaai/figs/functional/%d/syn_%d.pkl'%(i,j), 'wb'))
    else:
      print(force_closure[j], surface_distance[j], penetration[j])
