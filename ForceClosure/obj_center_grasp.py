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
from GraspPrediction import GraspPrediction
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel

np.seterr(all='raise')

# prepare argument
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_handcode', default=6, type=int)
parser.add_argument('--n_graspcode', default=10, type=int)
parser.add_argument('--n_iter', default=int(1e6), type=int)
parser.add_argument('--n_optim_iter', default=10, type=int)
parser.add_argument('--optim_stepsize', default=0.1, type=float)
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

object_model = ObjectModel(
  state_dict_path="/media/tengyu/2THDD/DeepSDF/DeepSDF/experiments/DeepSDF_antelope/ModelParameters/2000.pth"
)

fc_loss = FCLoss(object_model=object_model)
grasp_prediction = GraspPrediction(num_cp=3, hand_code_length=hand_model.code_length).cuda()
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

# prepare utilities
optimizer = torch.optim.Adam(
  grasp_prediction.parameters(),
  lr=args.lr, betas=(0.99, 0.999), eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch:args.lr_decay)
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

  # predict contact point, contact weight on hand, hand shape
  contact_point, contact_weight_heatmap, z = grasp_prediction(obj_code, grasp_code)
  contact_point_normal = object_model.gradient(contact_point, object_model.distance(obj_code, contact_point), retain_graph=True, create_graph=True)
  contact_weight_per_vert = hand_model.texture_color_per_vertex(contact_weight_heatmap)
  # cp: B x N x 3, cw: B x V x N, z: B x 10

  # # optimize z to minimize distance
  # z_prime = z.clone()
  # history = defaultdict(list)
  # for optim_iter in range(args.n_optim_iter):
  #   hand_verts = hand_model.get_vertices(z_prime)
  #   distance = torch.norm(contact_point.unsqueeze(1) - hand_verts.unsqueeze(2), dim=-1)
  #   weighted_distance = (distance * contact_weight_per_vert).mean((1,2))
  #   penetration = penetration_model.get_penetration(obj_code, z_prime) * 1e-2
  #   z_norm = fc_loss.l2_norm(z_prime[:,-args.n_handcode:])
  #   history['weighted_distance'].append(weighted_distance.detach().cpu().numpy())
  #   history['penetration'].append(penetration.detach().cpu().numpy())
  #   history['z_norm'].append(z_norm.detach().cpu().numpy())
  #   gradient = torch.autograd.grad((weighted_distance + penetration + z_norm).mean(), z_prime)[0]
  #   z_grad_ema.apply(gradient)
  #   z_prime = z_prime - gradient / z_grad_ema.average * args.optim_stepsize

  # hand_verts = hand_model.get_vertices(z_prime)
  # distance = torch.norm(contact_point.unsqueeze(1) - hand_verts.unsqueeze(2), dim=-1)
  # weighted_distance = (distance * contact_weight_per_vert).mean((1,2))
  # penetration = penetration_model.get_penetration(obj_code, z_prime) * 1e-2
  # z_norm = fc_loss.l2_norm(z_prime[:,-args.n_handcode:])
  # history['weighted_distance'].append(weighted_distance.detach().cpu().numpy())
  # history['penetration'].append(penetration.detach().cpu().numpy())
  # history['z_norm'].append(z_norm.detach().cpu().numpy())

  # history['weighted_distance'] = np.array(history['weighted_distance'])
  # history['penetration'] = np.array(history['penetration'])
  # history['z_norm'] = np.array(history['z_norm'])

  hand_verts = hand_model.get_vertices(z)
  distance = torch.norm(contact_point.unsqueeze(1) - hand_verts.unsqueeze(2), dim=-1)
  distance_zero_mean = distance - distance.mean(dim=1, keepdim=True)
  weighted_distance_for_z = (distance * contact_weight_per_vert.detach()).mean((1,2))
  weighted_distance_for_hm = (distance_zero_mean.detach() * contact_weight_per_vert).mean((1,2))
  penetration = penetration_model.get_penetration(obj_code, z)
  z_norm = fc_loss.l2_norm(z[:,-args.n_handcode:])

  # update with loss
  LinearIndependenceLoss, ForceClosureLoss, SurfaceDistanceLoss = map(torch.mean, fc_loss.fc_loss(contact_point, contact_point_normal, obj_code))
  WeightedHandDistanceZ = weighted_distance_for_z.mean()
  WeightedHandDistanceHM = weighted_distance_for_hm.mean()  
  # HandPredictionLoss = torch.norm(z - z_prime, dim=1).mean()
  HeatmapVariance = contact_weight_heatmap.var(dim=(0)).mean()
  loss = LinearIndependenceLoss + \
    ForceClosureLoss * 10 + \
      SurfaceDistanceLoss + \
        WeightedHandDistanceZ + \
          WeightedHandDistanceHM + \
            penetration.mean() * 1e-2 + \
              z_norm.mean() * 0.01


  # get heatmap gradient to visualize
  if args.viz:
    t0 = time.time()
    contact_hm_grad = torch.autograd.grad(loss, contact_weight_heatmap, retain_graph=True)[0]
    contact_hm_grad_vert = hand_model.texture_color_per_vertex(contact_hm_grad)
    contact_hm_grad_vert_plt = contact_hm_grad_vert - contact_hm_grad_vert.min(1,keepdim=True)[0]
    contact_hm_grad_vert_plt = contact_hm_grad_vert_plt / contact_hm_grad_vert_plt.max(1,keepdim=True)[0]
    viz_time = time.time() - t0

  # update weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  lr_scheduler.step(loss)

  # write summary
  writer.add_scalar('Metric/HeatmapVariance', HeatmapVariance.detach(), _iter)
  writer.add_scalar('Loss/LinearIndependenceLoss', LinearIndependenceLoss.detach(), _iter)
  writer.add_scalar('Loss/ForceClosureLoss', ForceClosureLoss.detach(), _iter)
  writer.add_scalar('Loss/SurfaceDistanceLoss', SurfaceDistanceLoss.detach(), _iter)
  writer.add_scalar('Loss/WeightedHandDistance', WeightedHandDistanceZ.detach(), _iter)
  writer.add_scalar('Loss/WeightedHandDistanceZeroMean', WeightedHandDistanceHM.detach(), _iter)
  # writer.add_scalar('Loss/HandPredictionLoss', HandPredictionLoss.detach(), _iter)
  writer.add_scalar('Loss/penetration', penetration.mean().detach(), _iter)
  writer.add_scalar('Loss/z_norm', z_norm.mean().detach(), _iter)
  writer.add_scalar('Loss/Total', loss.detach(), _iter)
  writer.add_scalar('LR', lr_scheduler._last_lr[0], _iter)
  writer.add_histogram('Gradient/contact_heatmap_vert', contact_hm_grad_vert.detach(), _iter)
  writer.add_histogram('Gradient/contact_heatmap', contact_hm_grad.detach(), _iter)
  print('\riter: %d loss: %f'%(_iter, loss), end='')
  
  # save model
  if (_iter+1) % 5000 == 0:
    torch.save(grasp_prediction.state_dict(), os.path.join(log_dir, 'weights', '%d.pth'%_iter))
    print()  

  if args.viz and _iter % 200 == 0:
    t0 = time.time()
    contact_point_numpy = contact_point.detach().cpu().numpy()
    contact_point_normal_numpy = contact_point_normal.detach().cpu().numpy()
    fig = plotly.tools.make_subplots(rows=2, cols=4, vertical_spacing=0.01, specs=[
      [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}], 
      [{'type': 'xy'}, {'type': 'xy'}, {'type': 'surface'}, {'type': 'surface'}]
    ], subplot_titles=[
      'weighted_distance', 'z norm', 'penetration', 'heatmap gradient', 'heatmap', 'heatmap gradient', 'before/after', 'heatmap in 3D'
    ])
    fig.append_trace(go.Histogram(x=weighted_distance_for_z.detach().cpu().numpy()), 1, 1)
    fig.append_trace(go.Histogram(x=z_norm.detach().cpu().numpy()), 1, 2)
    fig.append_trace(go.Histogram(x=penetration.detach().cpu().numpy()), 1, 3)

    # show histogram of contact heatmap gradients
    fig.append_trace(go.Histogram(x=contact_hm_grad_vert.detach().cpu().numpy().reshape([-1]), nbinsx=100), 1, 4)

    # draw heatmap
    cwpv = (contact_weight_per_vert[0].detach().cpu().numpy() * 255).astype(int)
    fig.append_trace(go.Scatter(
      x=hand_texture_coords[:,0], y=hand_texture_coords[:,1], mode='markers', marker=dict(line=dict(color='black', width=1), size=10, color=[f'rgb({cwpv[i,0]}, {cwpv[i,1]}, {cwpv[i,2]})' for i in range(len(cwpv))])
    ), 2, 1)

    # draw heatmap gradient
    chgvp = (contact_hm_grad_vert_plt[0].detach().cpu().numpy() * 255).astype(int)
    fig.append_trace(go.Scatter(
      x=hand_texture_coords[:,0], y=hand_texture_coords[:,1], mode='markers', marker=dict(line=dict(color='black', width=1), size=10, color=[f'rgb({chgvp[i,0]}, {chgvp[i,1]}, {chgvp[i,2]})' for i in range(len(chgvp))])
    ), 2, 2)

    # draw scene before and after optimization
    obj_mesh = get_obj_mesh(obj_idx[0])
    fig.append_trace(go.Mesh3d(
      x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], 
      color='lightblue', opacity=0.5
    ), 2, 3)
    hand_verts = hand_model.get_vertices(z[[0]]).detach().cpu().numpy()
    fig.append_trace(go.Mesh3d(
      x=hand_verts[0,:,0], y=hand_verts[0,:,1], z=hand_verts[0,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      color='lightpink', opacity=0.5
    ), 2, 3)
    # hand_verts = hand_model.get_vertices(z_prime[[0]]).detach().cpu().numpy()
    # fig.append_trace(go.Mesh3d(
    #   x=hand_verts[0,:,0], y=hand_verts[0,:,1], z=hand_verts[0,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
    #   color='red', opacity=0.5
    # ), 2, 3)
    for i in range(3):
      fig.append_trace(go.Cone(
        x=(contact_point_numpy[0,i,0], ), y=(contact_point_numpy[0,i,1], ), z=(contact_point_numpy[0,i,2], ), 
        u=(contact_point_normal_numpy[0,i,0], ), v=(contact_point_normal_numpy[0,i,1], ), w=(contact_point_normal_numpy[0,i,2], ),
        colorscale=[[0,'black'], [1, 'black']], showscale=False, sizemode='absolute', sizeref=0.5
      ), 2, 3)
    # draw heatmap on hand
    fig.append_trace(go.Mesh3d(
      x=hand_verts[0,:,0], y=hand_verts[0,:,1], z=hand_verts[0,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      vertexcolor=contact_weight_per_vert[0].detach().cpu().numpy()
    ), 2, 4)
    fig.append_trace(go.Cone(
      x=(contact_point_numpy[0,0,0], ), y=(contact_point_numpy[0,0,1], ), z=(contact_point_numpy[0,0,2], ), 
      u=(contact_point_normal_numpy[0,0,0], ), v=(contact_point_normal_numpy[0,0,1], ), w=(contact_point_normal_numpy[0,0,2], ),
      colorscale=[[0,'red'], [1, 'red']], showscale=False, sizemode='absolute', sizeref=0.5
    ), 2, 4)
    fig.append_trace(go.Cone(
      x=(contact_point_numpy[0,1,0], ), y=(contact_point_numpy[0,1,1], ), z=(contact_point_numpy[0,1,2], ), 
      u=(contact_point_normal_numpy[0,1,0], ), v=(contact_point_normal_numpy[0,1,1], ), w=(contact_point_normal_numpy[0,1,2], ),
      colorscale=[[0,'green'], [1, 'green']], showscale=False, sizemode='absolute', sizeref=0.5
    ), 2, 4)
    fig.append_trace(go.Cone(
      x=(contact_point_numpy[0,2,0], ), y=(contact_point_numpy[0,2,1], ), z=(contact_point_numpy[0,2,2], ), 
      u=(contact_point_normal_numpy[0,2,0], ), v=(contact_point_normal_numpy[0,2,1], ), w=(contact_point_normal_numpy[0,2,2], ),
      colorscale=[[0,'blue'], [1, 'blue']], showscale=False, sizemode='absolute', sizeref=0.5
    ), 2, 4)
    fig.update_layout(dict(scene=dict(aspectmode='data')), showlegend=False)

    f = open(os.path.join(log_dir, 'plotly.html'), 'w')
    f.write(fig.to_html())
    f.close()
