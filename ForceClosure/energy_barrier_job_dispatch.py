import pickle
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from typing import List, Any

basin_labels, basin_minima, basin_minima_energies, item_basin_barriers = pickle.load(open('adelm/ADELM_dispatch.pkl', 'rb'))

item_count = [0 for _ in range(len(basin_minima))]

for basin_label in basin_labels:
  item_count[basin_label] += 1

n_basin = len(basin_minima)
barriers = np.zeros([n_basin, n_basin]) + np.inf
barrier_indices = np.arange(len(barriers))

for i_item in range(len(item_basin_barriers)):
  i_basin = basin_labels[i_item]
  for j_basin in range(len(item_basin_barriers[i_item])):
    barriers[i_basin, j_basin] = min(item_basin_barriers[i_item][j_basin], barriers[i_basin, j_basin])
    if i_basin == j_basin:
      barriers[i_basin, j_basin] = basin_minima_energies[i_basin]

disconnected_basins = np.isinf(barriers).sum(1) == 86
barriers = barriers[~disconnected_basins][:,~disconnected_basins]
barrier_indices = barrier_indices[~disconnected_basins]
disconnected_basins = np.zeros(len(barriers)).astype(bool)
disconnected_basins[26] = True
barriers = barriers[~disconnected_basins][:,~disconnected_basins]
barrier_indices = barrier_indices[~disconnected_basins]

for i_basin in range(len(barriers)):
  for j_basin in range(len(barriers)):
    barriers[i_basin, j_basin] = min(barriers[i_basin, j_basin], barriers[j_basin, i_basin])


def consolidate(barriers):
  # input: N x N mat
  from scipy.sparse import csr_matrix
  from scipy.sparse.csgraph import floyd_warshall
  sparse_barrier = csr_matrix(barriers.shape)
  for i in range(len(barriers)):
    for j in range(len(barriers)):
      if not np.isinf(barriers[i,j]):
        sparse_barrier[i,j] = barriers[i,j]
  shortest_distance, shortest_path = floyd_warshall(sparse_barrier, directed=False, return_predecessors=True, unweighted=True)
  min_barrier = barriers.copy()
  for i in range(len(barriers)):
    for j in range(len(barriers)):
      path = []
      while j != i:
        path.append(j)
        j = shortest_path[i,j]
      path.append(i)
      path.reverse()
      __min_barrier = -np.inf
      for cur_id in range(len(path)-1):
        next_id = cur_id + 1
        __min_barrier = max(__min_barrier, barriers[path[cur_id], path[next_id]])
        min_barrier[i, path[next_id]] = min(__min_barrier, min_barrier[i, path[next_id]])
  return min_barrier

barriers = consolidate(barriers)

barriers_backup = barriers.copy()

# TODO: use ADELM method instead of the default methods in dendrogram()
# output: Z: N x 4: left-child, right-child, height, num-children
Z = []
total_examples = len(barriers)
for _iter in range(total_examples-1):
  # find next closest pair
  min_bar_idx = (barriers + np.eye(len(barriers)) * 1000 ).argmin()
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
  barriers = np.concatenate([barriers, barriers[[i,j]].min(0, keepdims=True)], axis=0)
  barriers = np.concatenate([barriers, barriers[:,[i,j]].min(1, keepdims=True)], axis=1)
  barriers[i,:] = np.inf
  barriers[j,:] = np.inf
  barriers[:,i] = np.inf
  barriers[:,j] = np.inf
  if i < total_examples:
    N = 1
  else:
    N = Z[i-total_examples][3]
  if j < total_examples:
    N += 1
  else:
    N += Z[j-total_examples][3]
  Z.append([i,j,B,N])

barriers = barriers_backup.copy()

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
  y0 = False
  y3 = False
  if y[0] == 0:
    i_leaf = int((x[0] - 5) / 10)
    y[0] = get_example_energy(R['leaves'][i_leaf])
    minima.append(y[0])
    y0 = True
    _ = ax.text(x[0], y[0] - 0.05, '%d(%d)'%(barrier_indices[R['leaves'][i_leaf]], item_count[barrier_indices[R['leaves'][i_leaf]]]), fontsize='xx-small')
    # for i_img, instance in enumerate(instance_ids_in_cluster):
    #   img = mpimg.imread('adelm_result/all/%d.png'%instance)
    #   imgbox = OffsetImage(img, zoom=img_zoom)
    #   ab = AnnotationBbox(imgbox, (x[0], y[0] - img_size - img_size * 2 * i_img))
    #   _ = ax.add_artist(ab)
    #   min_y = min(min_y, y[0] - img_size * 2 - img_size * 2 * i_img)
  if y[3] == 0:
    i_leaf = int((x[3] - 5) / 10)
    y[3] = get_example_energy(R['leaves'][i_leaf])
    minima.append(y[3])
    y3 = True
    _ = ax.text(x[3], y[3] - 0.05, '%d(%d)'%(barrier_indices[R['leaves'][i_leaf]], item_count[barrier_indices[R['leaves'][i_leaf]]]), fontsize='xx-small')
    # for i_img, instance in enumerate(instance_ids_in_cluster):
    #   img = mpimg.imread('adelm_result/all/%d.png'%instance)
    #   imgbox = OffsetImage(img, zoom=img_zoom)
    #   ab = AnnotationBbox(imgbox, (x[3], y[3] - img_size - img_size * 2 * i_img))
    #   _ = ax.add_artist(ab)
    #   min_y = min(min_y, y[3] - img_size * 2 - img_size * 2 * i_img)
  _ = ax.plot(x, y, c='k')
  if y0:
    _ = ax.scatter(x[0], y[0], s=item_count[barrier_indices[R['leaves'][i_leaf]]] * 10, facecolor='white', edgecolor='black')
  if y3:
    _ = ax.scatter(x[3], y[3], s=item_count[barrier_indices[R['leaves'][i_leaf]]] * 10, facecolor='white', edgecolor='black')

ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axes.get_xaxis().set_visible(False)
# fig.show()
fig.savefig('dendrogram_with_axis.png')

pickle.dump([Z, R, instance_ids], open('dendrogram.pkl', 'wb'))
# plt.show()

