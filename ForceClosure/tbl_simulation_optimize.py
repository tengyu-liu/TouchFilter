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

os.makedirs('optimize', exist_ok=True)
_id = sys.argv[1]
fn = 'logs/zeyu_5p/final_' + _id + '.pkl'

obj_code, z, contact_point_indices, energy, energy_history, temperature_history, stepsize_history = pickle.load(open(fn, 'rb'))

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

mask = torch.tensor(np.eye(15)).float().cuda().unsqueeze(0)  # 1 x 6 x 6
# mask = torch.cat([torch.zeros([1,6,9]).float().cuda(), mask], dim=2)

energy = compute_energy(obj_code, z, contact_point_indices, sd_weight=100)
old_energy = energy.clone()
grad = torch.autograd.grad(energy.sum(), z)[0]
grad_ema = EMA(0.98)
grad_ema.apply(grad)
mean_energy = []
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
    grad_ema.apply(grad)
    if _iter % 100 == 0:
        print(_iter, (energy-old_energy).mean().detach().cpu().numpy(), accept.float().mean().detach().cpu().numpy())

pickle.dump([obj_code, z, contact_point_indices], open(fn[:-4] + '_optim.pkl', 'wb'))
print(j)


