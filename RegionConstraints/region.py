import numpy as np
import pickle as pkl
import os
import torch
import plotly
from plotly import graph_objects as go
import trimesh as tm

from HandModel import HandModel
from ObjectModel import ObjectModel
import CodeUtil

hand_model = HandModel(flat_hand_mean=False)
object_model = ObjectModel()

def collate(Z):
  codes, zs, indices = [], [], []
  for obj_code, z, contact_point_indices in Z:
    codes.append(obj_code)
    zs.append(z)
    indices.append(contact_point_indices)
  return torch.stack(codes, 0), torch.stack(zs, 0), torch.stack(indices, 0)

Y, energies = pkl.load(open('../ForceClosure/aaai/supplementary/code_and_data/results/ADELM_proposals.pkl', 'rb'))

obj_code, z, contact_point_indices = collate(Y)
import matplotlib.pyplot as plt
plt.hist(d)
plt.show()

hand = hand_model.get_vertices(z).detach().cpu().numpy()
joints = hand_model.get_joints(z)
normals = hand_model.get_surface_normals(z=z).detach().cpu().numpy()

intersect = []

for i in range(len(Y)):
    print(i, '/', len(Y))
    obj = CodeUtil.get_obj_mesh_by_code(obj_code[i])
    intersection = tm.ray.ray_triangle.RayMeshIntersector(obj)
    intersect.append(intersection.intersects_any(hand[i], normals[i]))

affordance = np.stack(intersect, axis=0).mean(0)

fig = go.Figure(data=[
    go.Mesh3d(x=hand[i,:,0], y=hand[i,:,1], z=hand[i,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], intensity=affordance, opacity=1),
    # go.Mesh3d(x=obj.vertices[:,0], y=obj.vertices[:,1], z=obj.vertices[:,2], i=obj.faces[:,0], j=obj.faces[:,1], k=obj.faces[:,2], color='lightblue', opacity=0.5),
])
fig.show()
input()

np.save('grasp_region.npy', affordance)