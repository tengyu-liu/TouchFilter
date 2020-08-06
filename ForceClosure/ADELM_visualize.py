import os
import pickle

import numpy as np
import torch
from plotly import graph_objects as go

from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel
from CodeUtil import get_obj_mesh_by_code

# prepare models
hand_model = HandModel(
  n_handcode=6,
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False)

object_model = ObjectModel(
  state_dict_path="data/ModelParameters/2000.pth"
)

fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

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
  z_norm = torch.norm(z[:,-6:], dim=-1)
  if verbose:
    return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm, normal_alignment
  else:
    return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm + normal_alignment


def load_proposals(path):
  obj_code, z, contact_point_indices, energy, _, _, _ = pickle.load(open(path, 'rb'))
  linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code, z, contact_point_indices, verbose=True)
  fltr = ((force_closure < 0.1) * (surface_distance < 0.02) * (penetration < 0.02))
  Y = list(zip(obj_code[fltr], z[fltr], contact_point_indices[fltr]))
  return Y, energy.detach().cpu().numpy()

def inf():
  return float('inf')

data, energies = load_proposals('logs/saved_806000.pkl')
basin_labels, basin_minima, basin_minima_energies, energy_barriers, minimum_barrier_mc_chains = pickle.load(open('ADELM.pkl', 'rb'))
print(basin_labels)
print(basin_minima_energies)
print(energy_barriers)

# from plotly.subplots import make_subplots

# pair = [(3,5), (18, 78), (25,27), (32,30)]
# for i, j in pair:
#   fig = make_subplots(rows=1, cols=2, specs=[[{'type':'surface'}, {'type':'surface'}]])
#   obj_code_x, z_x, _ = data[i]
#   obj_code_y, z_y, _ = data[j]
#   mesh_x = get_obj_mesh_by_code(obj_code_x)
#   mesh_y = get_obj_mesh_by_code(obj_code_y)
#   hand_v_x = hand_model.get_vertices(z_x.unsqueeze(0))
#   hand_v_y = hand_model.get_vertices(z_y.unsqueeze(0))
#   sdx = object_model.distance(obj_code_x, hand_v_x)
#   sdy = object_model.distance(obj_code_y, hand_v_y)
#   hand_v_x = hand_v_x[0].detach().cpu().numpy()
#   hand_v_y = hand_v_y[0].detach().cpu().numpy()
#   fig.append_trace(go.Mesh3d(x=mesh_x.vertices[:,0], y=mesh_x.vertices[:,1], z=mesh_x.vertices[:,2], i=mesh_x.faces[:,0], j=mesh_x.faces[:,1], k=mesh_x.faces[:,2], color='lightblue', opacity=0.3), 1, 1)
#   fig.append_trace(go.Mesh3d(x=hand_v_x[:,0], y=hand_v_x[:,1], z=hand_v_x[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3), 1, 1)
#   fig.append_trace(go.Mesh3d(x=mesh_y.vertices[:,0], y=mesh_y.vertices[:,1], z=mesh_y.vertices[:,2], i=mesh_y.faces[:,0], j=mesh_y.faces[:,1], k=mesh_y.faces[:,2], color='lightblue', opacity=0.3), 1, 2)
#   fig.append_trace(go.Mesh3d(x=hand_v_y[:,0], y=hand_v_y[:,1], z=hand_v_y[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3), 1, 2)
#   print(torch.square(sdx - sdy).sum(1).squeeze())
#   fig.show()
#   input()

# exit()
  
os.makedirs('adelm_result', exist_ok=True)

for label, item in enumerate(basin_minima):
  os.makedirs('adelm_result/%d'%label, exist_ok=True)
  obj_code, z, cp_idx = item
  obj_mesh = get_obj_mesh_by_code(obj_code)
  cp_idx = cp_idx.detach().cpu().numpy()
  hand_verts = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  fig = go.Figure(data=[
      go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue', opacity=0.3), 
      go.Mesh3d(x=hand_verts[:,0], y=hand_verts[:,1], z=hand_verts[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3),
      go.Scatter3d(x=hand_verts[cp_idx, 0], y=hand_verts[cp_idx, 1], z=hand_verts[cp_idx,2], mode='markers', marker=dict(size=5, color='red'))
    ])
  fig.write_image('adelm_result/%d/minima.png'%(label))

for i in range(len(data)):
  obj_code, z, cp_idx = data[i]
  label = basin_labels[i]
  cp_idx = cp_idx.detach().cpu().numpy()
  obj_mesh = get_obj_mesh_by_code(obj_code)
  hand_verts = hand_model.get_vertices(z.unsqueeze(0))[0].detach().cpu().numpy()
  fig = go.Figure(data=[
      go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue', opacity=0.3), 
      go.Mesh3d(x=hand_verts[:,0], y=hand_verts[:,1], z=hand_verts[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3),
      go.Scatter3d(x=hand_verts[cp_idx, 0], y=hand_verts[cp_idx, 1], z=hand_verts[cp_idx,2], mode='markers', marker=dict(size=5, color='red'))
    ])
  fig.write_image('adelm_result/%d/%d.png'%(label, i))

hand_model2 = HandModel(flat_hand_mean=True, root_rot_mode='axisang')
z = torch.normal(0, 1, [1, 12], device='cuda').float() * 1e-6
hand_verts = hand_model2.get_vertices(z)[0].detach().cpu().numpy()

for i,j in minimum_barrier_mc_chains.keys():
  if minimum_barrier_mc_chains[(i,j)] is None or len(minimum_barrier_mc_chains[(i,j)]) < 3:
    continue
  os.makedirs('adelm_result/%d-%d'%(i,j), exist_ok=True)
  obj_mesh = get_obj_mesh_by_code(torch.tensor(minimum_barrier_mc_chains[(i,j)][0][0]).cuda())
  total = len(minimum_barrier_mc_chains[(i,j)])
  step = int(max(total // 40, 1))
  for k, state in enumerate(minimum_barrier_mc_chains[(i,j)][::step]):
    print(i,j,k, len(minimum_barrier_mc_chains[(i,j)]))
    obj_code, z, cp_idx = state
    hand_verts = hand_model2.get_vertices(torch.tensor(z).cuda())[0].detach().cpu().numpy()
    # cp_idx = cp_idx.detach().cpu().numpy()
    fig = go.Figure(data=[
      go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue', opacity=0.3), 
      go.Mesh3d(x=hand_verts[:,0], y=hand_verts[:,1], z=hand_verts[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3),
      go.Scatter3d(x=hand_verts[cp_idx, 0], y=hand_verts[cp_idx, 1], z=hand_verts[cp_idx,2], mode='markers', marker=dict(size=5, color='red'))
      ])
    # fig.update_layout(scene_camera=dict(
    #   up=dict(x=0, y=1, z=0)
    # ))
    fig.write_image('adelm_result/%d-%d/%d.png'%(i,j,k))
  k += 1
  state = minimum_barrier_mc_chains[(i,j)][-1]
  obj_code, z, cp_idx = state
  hand_verts = hand_model2.get_vertices(torch.tensor(z).cuda())[0].detach().cpu().numpy()
  # cp_idx = cp_idx.detach().cpu().numpy()
  fig = go.Figure(data=[
    go.Mesh3d(x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], color='lightblue', opacity=0.3), 
    go.Mesh3d(x=hand_verts[:,0], y=hand_verts[:,1], z=hand_verts[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink', opacity=0.3),
    go.Scatter3d(x=hand_verts[cp_idx, 0], y=hand_verts[cp_idx, 1], z=hand_verts[cp_idx,2], mode='markers', marker=dict(size=5, color='red'))
    ])
  # fig.update_layout(scene_camera=dict(
  #   up=dict(x=0, y=1, z=0)
  # ))
  fig.write_image('adelm_result/%d-%d/%d.png'%(i,j,k))
  os.system('ffmpeg -i adelm_result/%d-%d/%%d.png -y -filter_complex "[0:v] palettegen" adelm_result/%d-%d/palette.png'%(i,j,i,j))
  os.system('ffmpeg -i adelm_result/%d-%d/%%d.png -i adelm_result/%d-%d/palette.png -y -filter_complex "[0:v][1:v] paletteuse" adelm_result/%d-%d.gif'%(i,j,i,j,i,j))
  os.system('rm adelm/%d-%d/palette.png'%(i,j))

