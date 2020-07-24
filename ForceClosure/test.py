
import torch
import numpy as np

from HandModel import HandModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hand_model = HandModel(n_handcode=45)

z = np.random.random([1, hand_model.code_length])
z[0,9:] = 0
z[0,:3] = 0

plt.ion()
ax = plt.subplot(111, projection='3d')

for i in range(45):
  for j in np.linspace(-1,1,40):
    new_z = z.copy()
    new_z[:,9+i] = j
    verts = hand_model.get_vertices(torch.tensor(new_z).float().cuda()).detach().cpu().numpy()
    ax.cla()
    ax.scatter(verts[0,:,0], verts[0,:,1], verts[0,:,2])
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.pause(1e-5)
