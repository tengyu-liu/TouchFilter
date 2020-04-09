import numpy as np
import mayavi.mlab as mlab

from MagnetizedModel import MagnetizedModel
from utils.vis_util import VisUtil

batch_size = 16

model = MagnetizedModel(batch_size=batch_size)
vis_util = VisUtil(offscreen=False)

# Sample from energy function
sampled_x, energies = model.sample_from_U()
for i in range(batch_size):
  print('Energy: ', energies[i])
  vis_util.visualize(sampled_x[i])
  mlab.show()

# Compute distance between two states
x1 = model.sample_from_U()[0]
x2 = model.sample_from_U()[0]
distances = model.get_distance(x1, x2)
for i in range(batch_size):
  print('Distance: ', distances[i])
  vis_util.visualize(x1[i])
  mlab.show()
  vis_util.visualize(x2[i])
  mlab.show()
  break

# Magnetization
import matplotlib.pyplot as plt
plt.ion()
ax1 = plt.subplot(231)
ax2 = plt.subplot(232)
ax3 = plt.subplot(233)
ax4 = plt.subplot(234)
ax5 = plt.subplot(235)

jrot_dist_list = []
grot_dist_list = []
gpos_dist_list = []
distance_list = []
energy_list = []

for i in range(1000):
  grad = model.get_magnetized_gradient(x=x1, target=x2, alpha=0.001)
  x1 -= grad * 0.1 + np.random.uniform(np.zeros_like(x1), np.ones_like(x1)) * 0.0001
  jrot_dist, grot_dist, gpos_dist = model.get_distance(x1, x2)
  jrot_dist_list.append(jrot_dist.mean())
  grot_dist_list.append(grot_dist.mean())
  gpos_dist_list.append(gpos_dist.mean())
  distance_list.append((jrot_dist + grot_dist + gpos_dist * 10).mean())
  energy_list.append(model.get_energy(x1).mean())
  vis_util.visualize(x1[0])
  mlab.savefig('figs/%d.png'%i)
  ax1.cla()
  ax1.plot(jrot_dist_list)
  ax2.cla()
  ax2.plot(grot_dist_list)
  ax3.cla()
  ax3.plot(gpos_dist_list)
  ax4.cla()
  ax4.plot(distance_list)
  ax5.cla()
  ax5.plot(energy_list)
  plt.pause(1e-5)

plt.savefig('figs/energies.png')