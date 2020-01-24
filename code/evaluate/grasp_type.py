import copy
import os
import pickle
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import skimage.io as sio
import trimesh as tm
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion.quaternion import Quaternion as Q
from sklearn.cluster import *
from sklearn.decomposition import *
from sklearn.manifold import *

import mat_trans as mt
from forward_kinematics import ForwardKinematic

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

ffmpeg = 'ffmpeg'
if os.name == 'nt':
    ffmpeg = 'ffmpeg4'

def rotation_matrix(rot):
    a1 = rot[:,0]
    a2 = rot[:,1]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    eye = np.eye(4)
    eye[:3,:3] = np.stack([b1, b2, b3], axis=-1)
    return eye

cup_models = {i: tm.load_mesh('../../data/cups/onepiece/%d.obj'%i) for i in range(1,9)}
obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

def visualize(cup_id, hand_z):
    cup_model = cup_models[cup_id]
    mlab.triangular_mesh(cup_model.vertices[:,0], cup_model.vertices[:,1], cup_model.vertices[:,2], cup_model.faces, color=(0, 0, 1))
    z_ = hand_z
    jrot = z_[:22]
    grot = np.reshape(z_[22:28], [3, 2])
    gpos = z_[28:]
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))
    qpos = np.concatenate([gpos, grot, jrot])
    xpos, xquat = ForwardKinematic(qpos)
    obj_base = '../../data/hand'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
    for pid in range(4, 25):
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
            p.apply_translation(xpos[pid,:])
            mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
        except:
            continue

obj_base = '../../data/hand'
__zero_jrot = np.zeros([22])
__zero_grot = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
__zero_gpos = np.zeros([3])
__zero_grot = mt.quaternion_from_matrix(rotation_matrix(__zero_grot))
__zero_qpos = np.concatenate([__zero_gpos, __zero_grot, __zero_jrot])
__zero_xpos, __zero_xquat = ForwardKinematic(__zero_qpos)
__zero_stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
for pid in range(4, 25):
    p = copy.deepcopy(__zero_stl_dict[parts[pid - 4]])
    try:
        p.apply_transform(tm.transformations.quaternion_matrix(__zero_xquat[pid,:]))
        p.apply_translation(__zero_xpos[pid,:])
        __zero_stl_dict[parts[pid - 4]] = p
    except:
        continue

def visualize_distance(cup_id, hand_z, idx):
    cup_model = cup_models[cup_id]
    z_ = hand_z
    jrot = z_[:22]
    grot = np.reshape(z_[22:28], [3, 2])
    gpos = z_[28:]
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))
    qpos = np.concatenate([gpos, grot, jrot])
    xpos, xquat = ForwardKinematic(qpos)
    obj_base = '../../data/hand'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
    fig_data = []
    x, y, z, i, j, k, intensity = [], [], [], [], [], [], []
    count = 0
    fig_data = [go.Mesh3d(x=cup_model.vertices[:,0], y=cup_model.vertices[:,1], z=cup_model.vertices[:,2], \
                            i=cup_model.faces[:,0], j=cup_model.faces[:,1], k=cup_model.faces[:,2], color='lightpink')]
    for pid in range(4, 25):
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
            p.apply_translation(xpos[pid,:])
            x.append(__zero_stl_dict[parts[pid - 4]].vertices[:,0])
            y.append(__zero_stl_dict[parts[pid - 4]].vertices[:,1])
            z.append(__zero_stl_dict[parts[pid - 4]].vertices[:,2])
            intensity.append(-np.power(-tm.proximity.signed_distance(cup_model, p.vertices), 1/2))
            i.append(__zero_stl_dict[parts[pid - 4]].faces[:,0] + count)
            j.append(__zero_stl_dict[parts[pid - 4]].faces[:,1] + count)
            k.append(__zero_stl_dict[parts[pid - 4]].faces[:,2] + count)
            count += len(p.vertices)
            fig_data.append(go.Mesh3d(x=p.vertices[:,0], y=p.vertices[:,1], z=p.vertices[:,2], \
                                    i=p.faces[:,0], j=p.faces[:,1], k=p.faces[:,2], color='lightblue'))
        except:
            raise
    # Draw figure 1
    fig1 = go.Figure(data=fig_data)
    camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
    fig1.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    fig1.write_image('figs/plotly/%04d-0.png'%idx)
    # Draw figure 2
    x, y, z, i, j, k, intensity = map(np.hstack, [x, y, z, i, j, k, intensity])
    fig2 = go.Figure(data=go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensity, showscale=False))
    camera = dict(eye=dict(x=0, y=0, z=-3), up=dict(x=0, y=-1, z=0))
    fig2.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    fig2.write_image('figs/plotly/%04d-1.png'%idx)

for i in range(len(data['syn_z'])):
    visualize_distance(3, data['syn_z'][i], i)
    break


obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}
def get_pts(hand_z):
    z_ = hand_z
    jrot = z_[:22]
    grot = np.reshape(z_[22:28], [3, 2])
    gpos = z_[28:]
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))
    qpos = np.concatenate([gpos, grot, jrot])
    xpos, xquat = ForwardKinematic(qpos)
    pts = []
    for pid in range(4, 25):
        pts.append(np.matmul(Q(xquat[pid,:]).rotation_matrix, stl_dict[parts[pid - 4]].vertices.T).T + xpos[pid,:])
    return np.vstack(pts)

data = pickle.load(open('synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
keep_ids = (data['syn_e'] < 3).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
data['syn_z2'] /= np.linalg.norm(data['syn_z2'], axis=-1, keepdims=True)
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]


n_clusters = 4

tsne = TSNE()
kmeans = MeanShift()
cluster_ids = kmeans.fit_predict(data['syn_z'][:,:22])

w_2d = tsne.fit_transform(data['syn_w'])
# for i in range(n_clusters):
#     plt.scatter(w_2d[cluster_ids==i, 0], w_2d[cluster_ids==i, 1], s=1)

# plt.show()

z_2d = tsne.fit_transform(data['syn_z'][:,:22])
for i in range(n_clusters):
    plt.scatter(z_2d[cluster_ids==i,0], z_2d[cluster_ids==i,1], s=1)

plt.show()

z2_2d = tsne.fit_transform(data['syn_z2'])
plt.scatter(z2_2d[:,0], z2_2d[:,1], s=1)
plt.show()

# for i in range(n_clusters):
#     plt.scatter(z2_2d[cluster_ids==i, 0], z2_2d[cluster_ids==i, 1])

# plt.show()

# for i in range(n_clusters):
#     plt.subplot(2,2,i + 1)
#     plt.hist(data['syn_e'][cluster_ids==i, 0])

# plt.show()

# for i in range(n_clusters):
#     for j in np.where(cluster_ids == i)[0][:10]:
#         mlab.clf()
#         visualize(3, data['syn_z'][j])
#         mlab.savefig('cluster_%d_%d.png'%(i, j))

# cluster by weights doesn't really work. Next step is cluster by SDF distances. 

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,480))
plt.ion()
for _ in range(20):
    idx = np.random.randint(0, len(data['syn_z'])-1)
    mlab.clf()
    visualize(3, data['syn_z'][idx])
    mlab.savefig('figs/%d-0.png'%_)
    plt.clf()
    plt.scatter(z2_2d[:,0], z2_2d[:,1], s=1, c='blue')
    plt.scatter(z2_2d[idx,0], z2_2d[idx,1], c='red')
    plt.axis('off')
    plt.pause(1e-6)
    plt.savefig('figs/%d-1.png'%_)
    os.system('%s -i figs/%d-0.png -i figs/%d-1.png -filter_complex hstack -y figs/%d.png'%(ffmpeg, _, _, _))
    os.remove('figs/%d-0.png'%_)
    os.remove('figs/%d-1.png'%_)


# cluster by weights doesn't really work. Next step is cluster by SDF distances. 
dists = np.zeros([len(data['syn_z']), 17232])
cup_model = tm.load_mesh('../../data/cups/onepiece/3.obj')
for i in range(len(data['syn_z'])):
    hand_pts = get_pts(data['syn_z'][i])
    dists[i,:] = tm.proximity.signed_distance(cup_model, hand_pts)
    print('\r%d/%d'%(i, len(data['syn_z'])), end='')

np.save('synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300_syn_vert_d.npy', dists)
dist_2d = tsne.fit_transform(dists)
plt.scatter(dist_2d[:,0], dist_2d[:,1], s=1)
plt.show()

pca = PCA(n_components=10)
dist_pca = pca.fit_transform(dists)
dist_2d_pca = tsne.fit_transform(dist_pca)
plt.scatter(dist_2d_pca[:,0], dist_2d_pca[:,1], s=1)
plt.show()

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,530))
plt.ion()
# for _ in range(len(data['syn_z'])):
#     # idx = np.random.randint(0, len(data['syn_z'])-1)
#     idx = _
#     mlab.clf()
#     visualize(3, data['syn_z'][idx])
#     mlab.savefig('figs/src/%d-0.png'%_)
#     plt.clf()
#     plt.scatter(dist_2d[:,0], dist_2d[:,1], s=1, c='blue')
#     plt.scatter(dist_2d[idx,0], dist_2d[idx,1], c='red')
#     plt.axis('off')
#     plt.pause(1e-6)
#     plt.savefig('figs/src/%d-1.png'%_)
#     os.system('%s -i figs/src/%d-0.png -i figs/src/%d-1.png -filter_complex hstack -y figs/%d.png'%(ffmpeg, _, _, _))
#     # os.remove('figs/src/%d-0.png'%_)
#     # os.remove('figs/src/%d-1.png'%_)

def conflict(xy,x,y,size):
    return xy[0] > x-size*2 and xy[0] < x+size*2 and xy[1] > y-size*2 and xy[1] < y+size*2

size = 10
ax = plt.subplot(111)
ax.scatter(z2_2d[:,0], z2_2d[:,1], s=1, zorder=1)
existing_xy = []

for i in range(20):
    idx = np.random.randint(0, len(data['syn_z'])-1)
    x, y = z2_2d[idx]
    while any([conflict(xy, x, y, size) for xy in existing_xy]):
        idx = np.random.randint(0, len(data['syn_z'])-1)
        x, y = z2_2d[idx]
    existing_xy.append([x,y])
    im = ax.imshow(sio.imread('figs/src/%d-0.png'%idx), extent=([x-size,x+size,y-size,y+size]), zorder=2)
    # patch = patches.Rectangle((x-5,y-5),10,10)
    # im.set_clip_path(patch)

plt.xlim([-100,100])
plt.ylim([-100,100])

plt.show()
