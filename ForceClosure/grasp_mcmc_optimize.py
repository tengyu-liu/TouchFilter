import random
import sys
import pickle

import plotly
import plotly.graph_objects as go

from CodeUtil import *
from EMA import EMA
from HandModel import HandModel
from HCGraspPrediction import GraspPrediction
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

import matplotlib.pyplot as plt

name = sys.argv[1]
i_iter = int(sys.argv[2])

hand_model = HandModel(flat_hand_mean=False)
object_model = ObjectModel()
fc_loss_model = FCLoss(object_model=object_model)
grasp_prediction = GraspPrediction(num_cp=3, hand_code_length=hand_model.code_length, num_handpoint=hand_model.num_points).cuda()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

data = pickle.load(open('logs/%s/saved_%d.pkl'%(name, i_iter), 'rb'))

obj_code, z, contact_point_indices, energy = data

# i_item = np.argmin(energy.detach().cpu().numpy())
i_item = int(random.random() * len(z))

def compute_energy(obj_code, z, contact_point_indices, verbose=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(z.shape[0]), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)
  contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)

  hand_normal = hand_model.get_surface_normals(verts=hand_verts)
  closest_distances, closest_indices = torch.norm(hand_verts.unsqueeze(2) - contact_point.unsqueeze(1), dim=-1).min(1)
  closest_normals = torch.stack([hand_normal[torch.arange(z.shape[0]), closest_indices[:,i], :] for i in range(3)], dim=1)
  closest_normals = closest_normals / torch.norm(closest_normals, dim=-1, keepdim=True)    
  hand_loss = closest_distances.sum(1)
  normal_alignment = ((closest_normals * contact_normal).sum(-1) + 1).sum(-1)
  linear_independence, force_closure, surface_distance = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-6:], dim=-1)
  loss = hand_loss * 0.1 + linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm * 0.1 + normal_alignment
  if verbose:
    return hand_loss * 0.1, linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm * 0.1, normal_alignment
  else:
    return loss

def visualize(obj_code, contact_point_indices, z):
  obj_mesh = get_obj_mesh_by_code(obj_code[i_item])
  hand_verts = hand_model.get_vertices(z).detach().float().cuda()
  contact_point = torch.stack([hand_verts[torch.arange(hand_verts.shape[0]), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  hand_verts = hand_verts.detach().cpu().numpy()
  contact_point = contact_point.detach().cpu().numpy()
  fig = plotly.tools.make_subplots(1, 1, specs=[[{'type': 'surface'}]])
  fig.append_trace(go.Mesh3d(
    x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], 
    color='lightblue', opacity=0.5
  ), 1, 1)
  fig.append_trace(go.Mesh3d(
    x=hand_verts[i_item,:,0], y=hand_verts[i_item,:,1], z=hand_verts[i_item,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
    color='lightpink', opacity=0.5
  ), 1, 1)
  fig.append_trace(go.Scatter3d(
    x=contact_point[i_item,:,0], y=contact_point[i_item,:,1], z=contact_point[i_item,:,2], mode='markers'
  ), 1, 1)
  fig.show()

visualize(obj_code, contact_point_indices, z)

hand_loss, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code, z, contact_point_indices, verbose=True)
print('hand_loss', hand_loss[i_item].detach().cpu().numpy())
print('linear_independence', linear_independence[i_item].detach().cpu().numpy())
print('force_closure', force_closure[i_item].detach().cpu().numpy())
print('surface_distance', surface_distance[i_item].detach().cpu().numpy())
print('penetration', penetration[i_item].detach().cpu().numpy())
print('z_norm', z_norm[i_item].detach().cpu().numpy())
print('normal_alignment', normal_alignment[i_item].detach().cpu().numpy())
# import matplotlib.pyplot as plt
# energy = compute_energy(obj_code, z, contact_point_indices, verbose=True)
energy_entries = ['hand_loss', 'linear_independence', 'force_closure', 'surface_distance', 'penetration', 'z_norm', 'normal_alignment']
# for i in range(7):
#   plt.subplot(3,3,i+1)
#   plt.hist(energy[i].detach().cpu().numpy())
#   plt.title(energy_entries[i])

# plt.show()

energy_history = []
plt.ion()

for _iter in range(1000):
  energy = compute_energy(obj_code, z, contact_point_indices)
  energy_history.append(energy.sum().detach().cpu().numpy())
  grad = torch.autograd.grad(energy.sum(), z)[0]
  z = z - grad * 1e-4
  plt.clf()
  plt.plot(energy_history)
  plt.pause(1e-5)

# print(energy[i_item].detach().cpu().numpy())

visualize(obj_code, contact_point_indices, z)
