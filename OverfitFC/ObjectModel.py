import copy
import torch

from model import OverfitSDF

class ObjectModel:
  def __init__(self, state_dict_path):
    self.decoder = torch.load(state_dict_path)

  def distance(self, obj_idx, x):
    # obj_code: B
    # x: B x P x 3
    return self.decoder(x, obj_idx)

  def gradient(self, x, distance, retain_graph=False, create_graph=False, allow_unused=False):
    return torch.autograd.grad([distance.sum()], [x], retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]
  
  def closest_point(self, obj_idx, x):
    distance = self.distance(obj_idx, x)
    gradient = self.gradient(x, distance)
    normal = gradient.clone()
    count = 0
    while torch.abs(distance).mean() > 0.003 and count < 100:
      x = x - gradient * distance * 0.5
      distance = self.distance(obj_idx, x)
      gradient = self.gradient(x, distance)
      count += 1
    return x.detach(), normal


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D 
  import numpy as np
  from CodeUtil import *

  # TODO: test if obj model has negative values outside object (suspicious in fc_then_grasp code for errorneous penetration behavior)

  # plt.ion()
  ax = plt.subplot(111, projection='3d')

  obj_model = ObjectModel()
  latent_code = get_obj_code_random(1)
  x = torch.tensor(np.random.random([1,200000,3])*2-1, requires_grad=True).float().cuda() * 2
  distance = obj_model.distance(latent_code, x)
  p = distance[0,:,0] > 0
  n = distance[0,:,0] <= 0
  # ax.scatter(x[0,p,0].cpu().detach().numpy(), x[0,p,1].cpu().detach().numpy(), x[0,p,2].cpu().detach().numpy(), c='blue', s=1)
  ax.scatter(x[0,n,0].cpu().detach().numpy(), x[0,n,1].cpu().detach().numpy(), x[0,n,2].cpu().detach().numpy(), c='red', s=1)
  ax.set_xlim([-2,2])
  ax.set_ylim([-2,2])
  ax.set_zlim([-2,2])
  plt.show()

  # steps = 0

  # while torch.abs(distance).mean() > 3e-3:
  #   x = x - gradient * distance
  #   distance = obj_model.distance(latent_code, x)
  #   gradient = obj_model.gradient(x, distance)
  #   ax.cla()
  #   # p = distance[0,:,0] > 0
  #   # n = distance[0,:,0] <= 0
  #   ax.scatter(x[0,p,0].cpu().detach().numpy(), x[0,p,1].cpu().detach().numpy(), x[0,p,2].cpu().detach().numpy(), c='blue', s=1)
  #   # ax.scatter(x[0,n,0].cpu().detach().numpy(), x[0,n,1].cpu().detach().numpy(), x[0,n,2].cpu().detach().numpy(), c='red', s=1)
  #   ax.set_xlim([-1,1])
  #   ax.set_ylim([-1,1])
  #   ax.set_zlim([-1,1])
  #   plt.pause(1e-5)
  #   steps += 1

  # print(steps)