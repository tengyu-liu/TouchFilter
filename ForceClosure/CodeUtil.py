import os
import numpy as np
import torch
import torch.nn as nn
import trimesh

code_path = 'data/Reconstructions/2000/Codes/ShapeNetCore.v2/02876657'
mesh_path = 'data/Reconstructions/2000/Meshes/ShapeNetCore.v2/02876657'

codes = []
meshes = []

for fn in os.listdir(code_path):
  codes.append(torch.load(os.path.join(code_path, fn)).squeeze().float().cuda())
  meshes.append(trimesh.load(os.path.join(mesh_path, fn[:-3] + 'ply')))

codes = torch.stack(codes, 0)

def get_obj_code_random(batch_size, code_length=256):
  # code = torch.normal(mean=0, std=0.1, size=[batch_size, code_length]).float().cuda()
  idx = torch.randint(0, len(codes), size=[batch_size], device='cuda')
  return codes[idx], idx

def get_obj_mesh(idx) -> trimesh.Trimesh:
  return meshes[idx]

def get_obj_mesh_by_code(code) -> trimesh.Trimesh:
  for i, c in enumerate(codes):
    if torch.norm(code - c) < 1e-8:
      return meshes[i]

def get_grasp_code_random(batch_size, code_length):
  code = torch.normal(mean=0, std=1, size=[batch_size, code_length], device='cuda').float()
  return code

