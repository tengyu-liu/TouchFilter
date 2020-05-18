import copy
import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import trimesh as tm
from plotly.subplots import make_subplots
from pyquaternion.quaternion import Quaternion as Q

sys.path.append('../evaluate')

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
    cup_model = tm.load_mesh('../../data/cups/onepiece/3.obj')
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
    camera = dict(eye=dict(x=1, y=1, z=1))
    fig1.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
    fig1.write_image('figs/same_z_diff_z2_2/%04d-0.png'%idx)
    # Draw figure 2
    x, y, z, i, j, k, intensity = map(np.hstack, [x, y, z, i, j, k, intensity])
    fig2 = go.Figure(data=go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensity, showscale=False))
    camera = dict(eye=dict(x=0, y=0, z=-2), up=dict(x=0, y=-1, z=0))
    fig2.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
    fig2.write_image('figs/same_z_diff_z2_2/%04d-1.png'%idx)

data = pickle.load(open('synthesis/same_z_diff_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))
keep_ids = (data['syn_e'] < 10).reshape([-1])
data['syn_w'] = data['syn_w'][keep_ids, :]
data['syn_z'] = data['syn_z'][keep_ids, :]
data['syn_z2'] = data['syn_z2'][keep_ids, :]
data['syn_z2'] /= np.linalg.norm(data['syn_z2'], axis=-1, keepdims=True)
data['syn_e'] = data['syn_e'][keep_ids, :]
data['syn_p'] = data['syn_p'][keep_ids, :]

for i in range(len(data['syn_z'])):
    visualize_distance(3, data['syn_z'][i], i)
