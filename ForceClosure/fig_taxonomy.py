import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

basin_labels, basin_minima, basin_minima_energies, item_basin_barriers = pickle.load(open('adelm_7/ADELM_dispatch.pkl', 'rb'))
item_basin_barriers = np.array(item_basin_barriers)
basin_basin_barriers = np.zeros([len(basin_minima), len(basin_minima)]) + np.inf

for i, l in enumerate(basin_labels):
    for l2 in range(len(basin_minima)):
        if l == l2:
            continue
        basin_basin_barriers[l,l2] = min(basin_basin_barriers[l,l2], item_basin_barriers[i, l2])
        basin_basin_barriers[l2,l] = min(basin_basin_barriers[l,l2], item_basin_barriers[i, l2])

ids = np.arange(91)

basin_basin_barriers[~np.isinf(basin_basin_barriers)] = 1 / basin_basin_barriers[~np.isinf(basin_basin_barriers)]
basin_basin_barriers[np.isinf(basin_basin_barriers)] = -1
del_idx = np.where(np.isclose(basin_basin_barriers.sum(1), -91))[0]
basin_basin_barriers = np.delete(basin_basin_barriers, del_idx, 0)
basin_basin_barriers = np.delete(basin_basin_barriers, del_idx, 1)
ids = np.delete(ids, del_idx, 0)

node_coordinates = np.random.random([len(basin_basin_barriers), 2])
edge_weights_torch = torch.tensor(basin_basin_barriers, requires_grad=True).cuda()

plt.ion()

def update(i, node_coordinates):
    node_coordinates_torch = torch.tensor(node_coordinates, requires_grad=True).cuda()
    stepsize = 1
    for step in range(i):
        T = 0.9995 ** step * 10
        pairwise_distance = (node_coordinates_torch.unsqueeze(0) - node_coordinates_torch.unsqueeze(1)).norm(dim=2)
        loss = (pairwise_distance * edge_weights_torch).sum()
        grad = torch.autograd.grad(loss, node_coordinates_torch)[0]
        node_coordinates_torch2 = node_coordinates_torch - 0.5 * grad * stepsize * stepsize + torch.normal(mean=0,std=stepsize,size=node_coordinates_torch.shape,device='cuda')
        node_coordinates_torch2 = node_coordinates_torch2 - node_coordinates_torch2.mean()
        node_coordinates_torch2 = node_coordinates_torch2 / node_coordinates_torch2.std()
        pairwise_distance2 = (node_coordinates_torch2.unsqueeze(0) - node_coordinates_torch2.unsqueeze(1)).norm(dim=2)
        loss2 = (pairwise_distance2 * edge_weights_torch).sum()
        if loss2 < loss:
            node_coordinates_torch = node_coordinates_torch2
        else:
            alpha = torch.rand(1, device='cuda').float()
            accept = alpha < torch.exp((loss - loss2) / T)
            if accept:
                node_coordinates_torch = node_coordinates_torch2
        print('\r%d: %f'%(step, loss), end='')
    node_coordinates = node_coordinates_torch.detach().cpu().numpy()
    return node_coordinates

node_coordinates = update(5000, node_coordinates)

def find(_i):
    _i = list(ids).index(_i)
    _ = plt.clf()
    _ = plt.scatter(node_coordinates[:,0], node_coordinates[:,1], s=1, c='red')
    for i in range(len(basin_basin_barriers)):
        for j in range(len(basin_basin_barriers)):
            if basin_basin_barriers[i,j] > 0:
                _ = plt.plot(node_coordinates[[i,j],0], node_coordinates[[i,j],1], c='black', linewidth=0.1)
    _ = plt.text(node_coordinates[_i,0], node_coordinates[_i,1], str(ids[_i]))
    plt.axis('off')

# manually labeled grasp types
basin_type = {}
basin_type[0] = ['adduction grip', 'intermediate']
basin_type[48] = ['power disk']
basin_type[67] = ['precision quadpod']
basin_type[12] = ['stick', 'intermediate']
basin_type[83] = ['inferior pincer', 'precision']
basin_type[57] = ['tripod', 'quadpod', 'adduction grip', 'precision']
basin_type[41] = ['power ring']
basin_type[49] = ['precision disk']
basin_type[71] = ['power disk']
basin_type[42] = ['tripod', 'precision']
basin_type[44] = ['power sphere']
basin_type[53] = ['power small diameter']
basin_type[8] = ['power ring']
basin_type[10] = ['precision sphere']
basin_type[28] = ['precision sphere']
basin_type[23] = ['power ring']
basin_type[24] = ['palmal pinch', 'precision']
basin_type[7] = ['stick', 'intermediate']
basin_type[56] = ['lateral', 'intermediate']
basin_type[69] = ['precision sphere', 'power ring', 'inferior pincer']
basin_type[46] = ['tripod', 'precision sphere']
basin_type[40] = ['precision sphere', 'sphere 4-finger']
basin_type[18] = ['precision sphere', 'power ring']
basin_type[11] = ['power sphere', 'stick']
basin_type[63] = ['power sphere', 'power ring']
basin_type[17] = ['power ring', 'power sphere']
basin_type[26] = ['palmar pinch', 'tripod', 'precision sphere']
basin_type[22] = ['power ring']
basin_type[80] = ['power ring', 'power sphere']
basin_type[45] = ['power ring']
basin_type[21] = ['power ring']
basin_type[20] = ['power ring']
basin_type[5] = ['power ring']
basin_type[55] = ['power ring']
basin_type[62] = ['prismatic', 'precision']
basin_type[29] = ['tripod', 'precision sphere']
basin_type[39] = ['precision sphere']
basin_type[38] = ['precision sphere', 'quadpod']
basin_type[87] = ['precision sphere', 'parallel extension']
basin_type[4] = ['precision sphere', 'adduction grip']
basin_type[2] = ['precision sphere', 'precision disk']
basin_type[3] = ['tripod', 'quadpod', 'precision sphere']
basin_type[25] = ['precision sphere', 'parallel extension']
basin_type[32] = ['precision sphere']
basin_type[9] = ['tripod', 'writing tripod', 'precision']
basin_type[70] = ['tripod', 'precision sphere']
basin_type[6] = ['palmar pinch', 'precision']       # todo: some power rings may belong to palmar pinch
basin_type[13] = ['palmar pinch', 'precision']
basin_type[15] = ['adduction grip', 'intermediate']
basin_type[19] = ['large diameter', 'power']
basin_type[31] = ['tripod', 'quadpod', 'precision']
basin_type[33] = ['lateral', 'intermediate']
basin_type[34] = ['unlisted']
basin_type[35] = ['lateral', 'intermediate']
basin_type[36] = ['power ring']
basin_type[72] = ['adducted thumb', 'power']
basin_type[73] = ['small diameter', 'power']
basin_type[74] = ['tripod']
basin_type[75] = ['adduction grip', 'intermediate']
basin_type[76] = ['power ring']
basin_type[77] = ['power ring']
basin_type[78] = ['medium wrap', 'power']
basin_type[81] = ['light tool']
basin_type[82] = ['power sphere']
basin_type[85] = ['lateral', 'intermediate']
basin_type[86] = ['unlisted']
basin_type[88] = ['unlisted']
basin_type[89] = ['unlisted']
basin_type[90] = ['adduction grip']


plt.ioff()

scatter_size = 160

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
_ = ax.scatter(node_coordinates[:,0], node_coordinates[:,1], s=1, c='lightpink')
for i in range(len(basin_basin_barriers)):
    for j in range(len(basin_basin_barriers)):
        if basin_basin_barriers[i,j] > 0:
            _ = ax.plot(node_coordinates[[i,j],0], node_coordinates[[i,j],1], c='black', linewidth=np.sqrt(basin_basin_barriers[i,j]) * 0.3)

for term, c in [('power', 'red'), ('precision', 'green'), ('intermediate', 'orange')]:
    for i in range(len(basin_basin_barriers)):
        if ids[i] in basin_type and any(term in x for x in basin_type[ids[i]]):
            _ = ax.scatter(node_coordinates[i,0], node_coordinates[i,1], s=scatter_size, c=c, zorder=2)

ax.axis('off')
plt.show()
# fig.savefig('taxonomy-power-precision.png')



fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
_ = ax.scatter(node_coordinates[:,0], node_coordinates[:,1], s=1, c='lightpink')
for i in range(len(basin_basin_barriers)):
    for j in range(len(basin_basin_barriers)):
        if basin_basin_barriers[i,j] > 0:
            _ = ax.plot(node_coordinates[[i,j],0], node_coordinates[[i,j],1], c='black', linewidth=np.sqrt(basin_basin_barriers[i,j]) * 0.3)

for term, c in [('diameter', 'green'), ('wrap', 'green'), ('power disk', 'blue'), ('power sphere', 'red')]:
    for i in range(len(basin_basin_barriers)):
        if ids[i] in basin_type and any(term in x for x in basin_type[ids[i]]):
            _ = ax.scatter(node_coordinates[i,0], node_coordinates[i,1], s=scatter_size, c=c, zorder=2)

ax.axis('off')
plt.show()
# fig.savefig('taxonomy-within-power.png')


fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
_ = ax.scatter(node_coordinates[:,0], node_coordinates[:,1], s=1, c='lightpink')
for i in range(len(basin_basin_barriers)):
    for j in range(len(basin_basin_barriers)):
        if basin_basin_barriers[i,j] > 0:
            _ = ax.plot(node_coordinates[[i,j],0], node_coordinates[[i,j],1], c='black', linewidth=np.sqrt(basin_basin_barriers[i,j]) * 0.3)

for term, c in [('sphere', 'red'), ('pod', 'green')]:
    for i in range(len(basin_basin_barriers)):
        if ids[i] in basin_type and any(term in x for x in basin_type[ids[i]]):
            _ = ax.scatter(node_coordinates[i,0], node_coordinates[i,1], s=scatter_size, c=c, zorder=2)

ax.axis('off')
plt.show()
# fig.savefig('taxonomy-sphere-pod.png')

