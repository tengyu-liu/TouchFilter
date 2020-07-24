import time
import argparse
import os
import shutil
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.utils.tensorboard
from mpl_toolkits.mplot3d import Axes3D

from CodeUtil import *
from EMA import EMA
from HCGraspPrediction import GraspPrediction
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

np.seterr(all='raise')

# prepare argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--n_graspcode', default=64, type=int)
parser.add_argument('--n_iter', default=int(1e6), type=int)
parser.add_argument('--n_optim_iter', default=10, type=int)
parser.add_argument('--optim_stepsize', default=0.1, type=float)
parser.add_argument('--n_sample', default=100, type=int)
parser.add_argument('--name', default='exp', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_decay', default=1, type=float)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

log_dir = os.path.join('logs', args.name)
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'src'), exist_ok=True)
for fn in os.listdir('.'):
  if fn[-3:] == '.py':
    shutil.copy(fn, os.path.join(log_dir, 'src', fn))
f = open(os.path.join(log_dir, 'command.txt'), 'w')
f.write(' '.join(sys.argv))
f.close()

if args.viz:
  import matplotlib.pyplot as plt
  plt.ion()

  import plotly
  import plotly.graph_objects as go

# prepare models
hand_model = HandModel(
  n_handcode=args.n_handcode,
  root_rot_mode='ortho6d', 
  robust_rot=False,
  mano_path='/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/ForceClosure/third_party/manopth/mano/models')

hand_verts_eye = torch.tensor(np.eye(hand_model.num_points)).float().cuda() # 778 x 778

object_model = ObjectModel(
  state_dict_path="/media/tengyu/2THDD/DeepSDF/DeepSDF/experiments/DeepSDF_antelope/ModelParameters/2000.pth"
)

fc_loss_model = FCLoss(object_model=object_model)
grasp_prediction = GraspPrediction(num_cp=3, hand_code_length=hand_model.code_length, num_handpoint=hand_model.num_points).cuda()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# prepare utilities
# optimizer = torch.optim.Adam(
#   grasp_prediction.parameters(),
#   lr=args.lr, betas=(0.99, 0.999), eps=1e-8)
optimizer = torch.optim.SGD(grasp_prediction.parameters(), lr=args.lr, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000)

z_grad_ema = EMA(0.01)

writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)

def debug(msg=''):
  if args.debug:
    print(msg)

hand_texture_coords = (hand_model.texture_coords.detach().cpu().numpy().reshape([-1,2]) + 1) / 2 * 8 - 0.5

for _iter in range(args.n_iter):
  # Load object code
  obj_code, obj_idx = get_obj_code_random(args.batch_size)
  grasp_code = get_grasp_code_random(args.batch_size, args.n_graspcode)

  # predict hand-centered contact point and initial hand shape
  contact_point, z = grasp_prediction(obj_code, grasp_code)
  contact_distance = object_model.distance(obj_code, contact_point)
  contact_normal = object_model.gradient(contact_point, contact_distance, retain_graph=True, create_graph=True)
  hand_verts = hand_model.get_vertices(z)
  hand_normal = hand_model.get_surface_normals(verts=hand_verts)
  
  closest_distances, closest_indices = torch.norm(hand_verts.unsqueeze(2) - contact_point.unsqueeze(1), dim=-1).min(1)
  closest_normals = torch.stack([hand_normal[torch.arange(args.batch_size), closest_indices[:,i], :] for i in range(3)], dim=1)

  hand_loss = closest_distances.sum(1)
  normal_alignment = (closest_normals * contact_normal).sum((-1,-2))
  linear_independence, force_closure, surface_distance = fc_loss_model.fc_loss(contact_point, contact_normal, obj_code)
  penetration = penetration_model.get_penetration_from_verts(obj_code, hand_verts)  # B x V
  z_norm = torch.norm(z[:,-args.n_handcode:], dim=-1)
  loss = (hand_loss * 0.1 + linear_independence + force_closure + surface_distance + penetration.sum(1) * 1e-3 + z_norm * 0.1 + normal_alignment).mean()

  # update weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  lr_scheduler.step(loss)

  # write summary
  writer.add_scalar('Loss/hand_loss', hand_loss.mean().detach(), _iter)
  writer.add_scalar('Loss/normal_alignment', normal_alignment.mean().detach(), _iter)
  writer.add_scalar('Loss/linear_independence', linear_independence.mean().detach(), _iter)
  writer.add_scalar('Loss/force_closure', force_closure.mean().detach(), _iter)
  writer.add_scalar('Loss/surface_distance', surface_distance.mean().detach(), _iter)
  writer.add_scalar('Loss/penetration', penetration.mean().detach(), _iter)
  writer.add_scalar('Loss/z_norm', z_norm.mean().detach(), _iter)
  writer.add_scalar('Loss/loss', loss.detach(), _iter)
  writer.add_scalar('LR', lr_scheduler._last_lr[0], _iter)
  print('\riter: %d loss: %f'%(_iter, loss), end='')
  
  # save model
  if (_iter+1) % 5000 == 0:
    torch.save(grasp_prediction.state_dict(), os.path.join(log_dir, 'weights', '%d.pth'%_iter))
    print()  

  if args.viz and _iter % 200 == 0:
    t0 = time.time()
    contact_point_numpy = contact_point.detach().cpu().numpy()
    contact_normal_numpy = contact_normal.detach().cpu().numpy()
    fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.01, specs=[
      [{'type': 'surface'}, {'type': 'surface'}]
    ], subplot_titles=[
      'scene', 'penetration'
    ])

    # draw scene before and after optimization
    obj_mesh = get_obj_mesh(obj_idx[0])
    hand_verts = hand_verts.detach().cpu().numpy()
    fig.append_trace(go.Mesh3d(
      x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], 
      color='lightblue', opacity=0.5
    ), 1, 1)
    fig.append_trace(go.Mesh3d(
      x=hand_verts[0,:,0], y=hand_verts[0,:,1], z=hand_verts[0,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      color='lightpink', opacity=0.5
    ), 1, 1)

    fig.append_trace(go.Scatter3d(
      x=(contact_point_numpy[0,0,0], ), y=(contact_point_numpy[0,0,1], ), z=(contact_point_numpy[0,0,2], ), 
      mode='markers', marker=dict(color='red', size=3)
    ), 1, 1)
    fig.append_trace(go.Scatter3d(
      x=(contact_point_numpy[0,1,0], ), y=(contact_point_numpy[0,1,1], ), z=(contact_point_numpy[0,1,2], ), 
      mode='markers', marker=dict(color='green', size=3)
    ), 1, 1)
    fig.append_trace(go.Scatter3d(
      x=(contact_point_numpy[0,2,0], ), y=(contact_point_numpy[0,2,1], ), z=(contact_point_numpy[0,2,2], ), 
      mode='markers', marker=dict(color='blue', size=3)
    ), 1, 1)

    # draw heatmap on hand
    fig.append_trace(go.Mesh3d(
      x=hand_verts[0,:,0], y=hand_verts[0,:,1], z=hand_verts[0,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      intensity=penetration[0].detach().cpu().numpy()
    ), 1, 2)

    fig.update_layout(dict(scene=dict(aspectmode='data')), showlegend=False)

    f = open(os.path.join(log_dir, 'plotly.html'), 'w')
    f.write(fig.to_html())
    f.close()
