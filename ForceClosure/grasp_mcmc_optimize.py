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

name = sys.argv[1]

hand_model = HandModel(flat_hand_mean=False)
object_model = ObjectModel()
fc_loss_model = FCLoss(object_model=object_model)
grasp_prediction = GraspPrediction(num_cp=3, hand_code_length=hand_model.code_length, num_handpoint=hand_model.num_points).cuda()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

data = pickle.load(open('logs/%s/last.pkl'%(name), 'rb'))

obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history = data

# i_item = np.argmin(energy.detach().cpu().numpy())
# i_item = int(random.random() * len(z))
# i_item = 53
# print(i_item)

def compute_energy(obj_code, z, contact_point_indices, verbose=False, no_grad=False):
  hand_verts = hand_model.get_vertices(z)
  contact_point = torch.stack([hand_verts[torch.arange(z.shape[0]), contact_point_indices[:,i],:] for i in range(3)], dim=1)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance)

  grad_op = torch.enable_grad
  if no_grad:
    grad_op = torch.no_grad

  with grad_op():
    contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)

    hand_normal = hand_model.get_surface_normals(verts=hand_verts)
    hand_normal = torch.stack([hand_normal[torch.arange(z.shape[0]), contact_point_indices[:,i], :] for i in range(3)], dim=1)
    hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    

    normal_alignment = ((hand_normal * contact_normal).sum(-1) + 1).sum(-1)
    linear_independence, force_closure, surface_distance = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
    penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
    z_norm = torch.norm(z[:,-6:], dim=-1)
    loss = linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + z_norm * 0.001 + normal_alignment
    if verbose:
      return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), z_norm * 0.001, normal_alignment
    else:
      return loss

# def visualize(obj_code, contact_point_indices, z, i_item):
#   obj_mesh = get_obj_mesh_by_code(obj_code[i_item])
#   hand_verts = hand_model.get_vertices(z).detach().float().cuda()
#   contact_point = torch.stack([hand_verts[torch.arange(hand_verts.shape[0]), contact_point_indices[:,i],:] for i in range(3)], dim=1)
#   hand_verts = hand_verts.detach().cpu().numpy()
#   contact_point = contact_point.detach().cpu().numpy()
#   fig = plotly.tools.make_subplots(1, 1, specs=[[{'type': 'surface'}]])
#   fig.append_trace(go.Mesh3d(
#     x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], 
#     color='lightblue', opacity=0.5
#   ), 1, 1)
#   fig.append_trace(go.Mesh3d(
#     x=hand_verts[i_item,:,0], y=hand_verts[i_item,:,1], z=hand_verts[i_item,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
#     color='lightpink', opacity=0.5
#   ), 1, 1)
#   fig.append_trace(go.Scatter3d(
#     x=contact_point[i_item,:,0], y=contact_point[i_item,:,1], z=contact_point[i_item,:,2], mode='markers'
#   ), 1, 1)
#   fig.show()

# visualize(obj_code, contact_point_indices, z, i_item)

# import matplotlib.pyplot as plt
energy = compute_energy(obj_code, z, contact_point_indices, verbose=True)
energy_entries = ['linear_independence', 'force_closure', 'surface_distance', 'penetration', 'z_norm', 'normal_alignment']
# for i in range(6):
#   # plt.subplot(3,3,i+1)
#   # plt.hist(energy[i].detach().cpu().numpy())
#   # plt.title(energy_entries[i])
#   print(energy_entries[i], energy[i][i_item].detach().cpu().numpy())

# plt.show()

# print('hand_loss', hand_loss[i_item].detach().cpu().numpy())
# print('linear_independence', linear_independence[i_item].detach().cpu().numpy())
# print('force_closure', force_closure[i_item].detach().cpu().numpy())
# print('surface_distance', surface_distance[i_item].detach().cpu().numpy())
# print('penetration', penetration[i_item].detach().cpu().numpy())
# print('z_norm', z_norm[i_item].detach().cpu().numpy())
# print('normal_alignment', normal_alignment[i_item].detach().cpu().numpy())

# plt.show()

# energy_history = [[] for _ in range(6)]

# plt.ion()

# obj_code = obj_code[[i_item, i_item-1]]
# z = z[[i_item, i_item]]
# contact_point_indices = contact_point_indices[[i_item, i_item-1]]

# print(z[0,-6:])

ema = EMA(0.05)
# ax = [plt.subplot(2,3,i+1) for i in range(6)]

for _iter in range(5000):
  for idx in [np.arange(0,128), np.arange(128,256)]:
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = compute_energy(obj_code[idx], z[idx], contact_point_indices[idx], verbose=True)
    grad = torch.autograd.grad((force_closure + surface_distance + penetration).sum(), z[idx], retain_graph=True)[0]
    z[idx,-6:] = z[idx,-6:] - grad[:,-6:] * 2e-2
    # ema.apply(grad)
    # grad = grad / ema.average
    # z = z - grad * 1e-5
    # energy_history[0].append(linear_independence[0].detach().cpu().numpy())
    # energy_history[1].append(force_closure[0].detach().cpu().numpy())
    # energy_history[2].append(surface_distance[0].detach().cpu().numpy())
    # energy_history[3].append(penetration[0].detach().cpu().numpy())
    # energy_history[4].append(z_norm[0].detach().cpu().numpy())
    # energy_history[5].append(normal_alignment[0].detach().cpu().numpy())
    # if _iter % 100 == 0 or _iter == 4999:
    #   for i in range(6):
    #     ax[i].cla()
    #     ax[i].plot(energy_history[i])
    #     ax[i].set_title(energy_entries[i])
    #   plt.pause(1e-5)
    print(_iter, linear_independence.detach().cpu().numpy().mean(), force_closure.detach().cpu().numpy().mean(), surface_distance.detach().cpu().numpy().mean(), penetration.detach().cpu().numpy().mean(), z_norm.detach().cpu().numpy().mean(), normal_alignment.detach().cpu().numpy().mean())

# print(energy[i_item].detach().cpu().numpy())
energy = compute_energy(obj_code, z, contact_point_indices, verbose=True)
energy_entries = ['linear_independence', 'force_closure', 'surface_distance', 'penetration', 'z_norm', 'normal_alignment']

for i in range(6):
  # plt.subplot(3,3,i+1)
  # plt.hist(energy[i].detach().cpu().numpy())
  # plt.title(energy_entries[i])
  print(energy_entries[i], energy[i][0].detach().cpu().numpy())

pickle.dump([obj_code, z, contact_point_indices, energy], open('logs/%s/optimized.pkl'%(name), 'wb'))
# visualize(obj_code, contact_point_indices, z, 0)
# input()