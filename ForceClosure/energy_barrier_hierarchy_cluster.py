import pickle
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from typing import List, Any

barriers = np.load('barrier.npy')
total_examples = len(barriers)


# TODO: use ADELM method instead of the default methods in dendrogram()
# output: Z: N x 4: left-child, right-child, height, num-children
Z = []
for _iter in range(total_examples-1):
  # find next closest pair
  min_bar_idx = (barriers + np.eye(len(barriers)) * 1000).argmin()
  i, j = min_bar_idx // len(barriers), min_bar_idx % len(barriers)
  # find which one is the new cluster minima
  i_energy = barriers[i,i]
  j_energy = barriers[j,j]
  B = barriers[i,j]
  minima_energy = min(i_energy, j_energy)
  minima_idx = i
  if i_energy > j_energy:
    minima_idx = j
  # append new cluster to barriers and set the two selected entries to inf
  barriers = np.concatenate([barriers, barriers[[minima_idx]]], axis=0)
  barriers = np.concatenate([barriers, barriers[:,[minima_idx]]], axis=1)
  barriers[i,:] = np.inf
  barriers[j,:] = np.inf
  barriers[:,i] = np.inf
  barriers[:,j] = np.inf
  if i < 131:
    N = 1
  else:
    N = Z[i-131][3]
  if j < 131:
    N += 1
  else:
    N += Z[j-131][3]
  Z.append([i,j,B,N])

barriers = np.load('barrier.npy')

def collect_instances(cluster_id):
  if cluster_id < total_examples:
    return [cluster_id]
  left, right, energy, num_nodes = Z[cluster_id - total_examples]
  result = collect_instances(int(left)) + collect_instances(int(right))
  assert len(result) == num_nodes
  return result

def get_example_energy(i):
  return barriers[i,i]

# draw image
R = dendrogram(Z, leaf_font_size=12, show_leaf_counts=True, no_plot=True)

instance_ids = []
for cluster_i in range(len(R['ivl'])):
  cluster_id = R['leaves'][cluster_i]
  instance_ids.append(collect_instances(cluster_id))

xs = np.array(R['icoord']) 
ys = np.array(R['dcoord'])
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
minima = []
i_leaf = 0
min_y = 0
for i in range(len(R['ivl'])-1):
  x = xs[i]
  y = ys[i]
  if ys[i][0] == 0:
    instance_ids_in_cluster = instance_ids[i_leaf]
    y[0] = get_example_energy(instance_ids_in_cluster).min()
    minima.append(y[0])
    _ = ax.scatter(x[0], y[0], s=len(instance_ids_in_cluster)*20, facecolor='white', edgecolor='black')
    _ = ax.text(x[0], y[0] - 2, R['leaves'][i_leaf])
    i_leaf += 1
    # for i_img, instance in enumerate(instance_ids_in_cluster):
    #   img = mpimg.imread('adelm_result/all/%d.png'%instance)
    #   imgbox = OffsetImage(img, zoom=img_zoom)
    #   ab = AnnotationBbox(imgbox, (x[0], y[0] - img_size - img_size * 2 * i_img))
    #   _ = ax.add_artist(ab)
    #   min_y = min(min_y, y[0] - img_size * 2 - img_size * 2 * i_img)
  if ys[i][3] == 0:
    instance_ids_in_cluster = instance_ids[i_leaf]
    y[3] = get_example_energy(instance_ids_in_cluster).min()
    minima.append(y[3])
    _ = ax.scatter(x[3], y[3], s=len(instance_ids_in_cluster)*20, facecolor='white', edgecolor='black')
    _ = ax.text(x[3], y[3] - 2, R['leaves'][i_leaf])
    i_leaf += 1
    # for i_img, instance in enumerate(instance_ids_in_cluster):
    #   img = mpimg.imread('adelm_result/all/%d.png'%instance)
    #   imgbox = OffsetImage(img, zoom=img_zoom)
    #   ab = AnnotationBbox(imgbox, (x[3], y[3] - img_size - img_size * 2 * i_img))
    #   _ = ax.add_artist(ab)
    #   min_y = min(min_y, y[3] - img_size * 2 - img_size * 2 * i_img)
  _ = ax.plot(x, y, c='k')

ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axes.get_xaxis().set_visible(False)
# fig.show()
fig.savefig('dendrogram.png')

pickle.dump([Z, R, instance_ids], open('dendrogram.pkl', 'wb'))
# plt.show()

# 1. [1, 103, 119, 102, 15, 74, 79, 111, 123, 93, 54, 113, 37, 11, 95]
# 2. [78, 122, 18, 108, 0, 71, 89, 115, 121, 9, 46, 106, 56, 59, 114, 118, 61, 98, 2]
# 3. [13, 42, 23, 48, 64, 112, 84, 94, 63, 127, 67, 14, 126, 28, 17, 129, 82, 25, 83, 101, 105, 90, 76, 7, 69, 10, 47, 117, 6, 19, 86, 128, 77, 34, 55, 5, 72, 97, 70, 120, 30, 32, 109, 51, 31]
# 4. [8, 39, 91, 16, 92, 68, 73, 22, 62, 53, 110, 104, 130, 36, 65, 35, 40, 12, 44, 50, 124, 80, 49, 81, 99, 125, 4, 24, 100,96, 107, 26, 116, 29, 41, 57, 66, 38, 58, 87, 85, 60, 88, 75, 45, 33, 27, 52, 21, 20, 3, 43]

[13, 42, 23, 48, 64, 112, 84, 94, 63, 127, 67, 14, 126]
[128, 77, 34, 55, 5, 72, 97, 70, 120, 30, 32]
[109]
[80, 49, 81, 99, 125,4, 24, 100, 96, 107, 26, 116, 29, 41, 57, 66, 38, 58, 87, 85, 60, 88]