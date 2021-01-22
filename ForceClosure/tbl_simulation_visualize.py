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
  contact_point = torch.stack([hand_verts[torch.arange(1), contact_point_indices[:,i],:] for i in range(5)], dim=1)
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

if os.path.exists('logs/rerun/final_optim.pkl'):
  obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = pickle.load(open('logs/rerun/final_optim.pkl', 'rb'))
else:
  obj_code, z, contact_point_indices = [], [], []
  linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = [],[],[],[],[],[]
  for _id in [0,1,2,3,4,5,7]:
    fn = 'logs/rerun/optim_%d.pkl'%_id
    _obj_code, _z, _contact_point_indices = pickle.load(open(fn, 'rb'))
    fltr = []
    for i in range(len(_z)):
      print('\r%d:%d'%(_id,i), end='')
      # print(_contact_point_indices[i])
      _linear_independence, _force_closure, _surface_distance, _penetration, _z_norm, _normal_alignment = compute_energy(_obj_code[[i]], _z[[i]], _contact_point_indices[[i]], sd_weight=1, verbose=True)
      if _force_closure.sum() < 1e-5 and _linear_independence < 1e-5 and _surface_distance.sum() < 0.01 and _penetration.sum() < 0.01:
        obj_code.append(_obj_code[i].detach())
        z.append(_z[i].detach())
        contact_point_indices.append(_contact_point_indices[i].detach())
        linear_independence.append(_linear_independence[0].detach())
        force_closure.append(_force_closure[0].detach())
        surface_distance.append(_surface_distance[0].detach())
        penetration.append(_penetration[0].detach())
        z_norm.append(_z_norm[0].detach())
        normal_alignment.append(_normal_alignment[0].detach())
  obj_code = torch.stack(obj_code, dim=0)
  z = torch.stack(z, dim=0)
  contact_point_indices = torch.stack(contact_point_indices, dim=0)
  pickle.dump([obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment], open('logs/rerun/final_optim.pkl', 'wb'))

print(len(z))

obj_code = obj_code.squeeze()
z = z.squeeze()
contact_point_indices = contact_point_indices.squeeze()
print(obj_code.shape, z.shape, contact_point_indices.shape)
hand_verts = hand_model.get_vertices(z).detach().cpu().numpy()
cpi = contact_point_indices.detach().cpu().numpy()

for i in range(len(z)):
  # if i != 180 and i!= 232:
  #   continue
  print(i, '/', len(z))
  print('\tlinear_independence', linear_independence[i])
  print('\tforce_closure', force_closure[i])
  print('\tsurface_distance', surface_distance[i])
  print('\tpenetration', penetration[i])
  print('\tz_norm', z_norm[i])
  print('\tnormal_alignment', normal_alignment[i])
  mesh = get_obj_mesh_by_code(obj_code[i])
  cp = hand_verts[i,cpi[i],:]
  fig = go.Figure(data=[
    go.Mesh3d(x=hand_verts[i,:,0], y=hand_verts[i,:,1], z=hand_verts[i,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], color='lightpink'),
    go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2], i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2], color='lightblue'),
    go.Scatter3d(x=cp[:,0], y=cp[:,1], z=cp[:,2], mode='markers', marker=dict(size=5, color='red'))
  ])
  fig.show()
  input()