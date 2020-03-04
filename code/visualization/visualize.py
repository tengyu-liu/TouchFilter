import copy
import os
import sys
import pickle

import numpy as np
import trimesh as tm
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mat_trans as mt
from forward_kinematics import ForwardKinematic

import matplotlib.pyplot as plt
from pyquaternion.quaternion import Quaternion as Q

name = sys.argv[1]
epoch = int(sys.argv[2])
batch = int(sys.argv[3])

mlab.options.offscreen = True

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

def visualize_hand(fig, weights, rows, i):
    xpos, xquat = ForwardKinematic(np.zeros([53]))

    obj_base = '../../data'
    stl_dict = {obj: np.load(os.path.join(obj_base, '%s.sample_points.npy'%obj)) for obj in parts}

    start = 0
    end = 0

    if len(weights.shape) == 2:
        weights = weights[:,0]

    for pid in range(4, 25):
        if '0' in parts[pid - 4]:
            continue
        p = copy.deepcopy(stl_dict[parts[pid - 4]])
        try:
            end += len(p)

            p = np.matmul(Q().rotation_matrix, p.T).T
            p += xpos[[pid], :]

            ax = fig.add_subplot(rows, 2, i * 2 - 1)
            pts = p[:,2] > 0.001
            ax.scatter(p[pts,0], p[pts,1], c=weights[start:end][pts])
            ax.axis('off')

            ax = fig.add_subplot(rows, 2, i * 2)
            pts = p[:,2] <= 0.001
            ax.scatter(p[pts,0], p[pts,1], c=weights[start:end][pts])
            ax.axis('off')

            start += len(p)
        except:
            raise
            continue

data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../figs', name, '%04d-%d.pkl'%(epoch, batch)), 'rb'))

cup_id = data['cup_id']
obs_z = np.array(data['obs_z'])
obs_e = data['obs_e']
obs_w = np.array(data['obs_w'])
syn_e = np.array(data['syn_e'])
syn_z = np.array(data['syn_z'])
syn_w = np.array(data['syn_w'])

fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
mlab.figure(size=(640,480))

for i_batch in range(len(syn_z)):
    mlab.clf()
    visualize(cup_id, obs_z[i_batch])
    mlab.savefig('../figs/%s/%04d-%d-%d.png'%(name, epoch, batch, i_batch))

    for i_seq in [90]:
        # Draw 3D grasping
        mlab.clf()
        visualize(cup_id, syn_z[i_batch][i_seq])
        mlab.savefig('../figs/%s/%04d-%d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq, 1))
        # Draw feature selection map
        fig.clf()
        if len(obs_w) == 1:
            visualize_hand(fig, obs_w[0], 2, 1)
            visualize_hand(fig, syn_w[0, i_seq], 2, 2)
        else:
            visualize_hand(fig, obs_w[i_batch], 2, 1)
            visualize_hand(fig, syn_w[i_batch, i_seq], 2, 2)
        ax = fig.add_subplot(221)
        ax.set_title('obs back')
        ax = fig.add_subplot(222)
        ax.set_title('obs front')
        ax = fig.add_subplot(223)
        ax.set_title('syn back')
        ax = fig.add_subplot(224)
        ax.set_title('syn front')
        fig.savefig('../figs/%s/%04d-%d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq, 2))
        # Draw energy plot
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(syn_e[i_batch, :i_seq])
        ax.plot([obs_e[i_batch] for _ in range(i_seq)])
        fig.savefig('../figs/%s/%04d-%d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq, 3))
        # Merge two
        os.system(ffmpeg + ' -i ../figs/%s/%04d-%d-%d-%d-%d.png -i ../figs/%s/%04d-%d-%d-%d-%d.png -filter_complex hstack ../figs/%s/%04d-%d-%d-%d-%d.png'%(
            name, epoch, batch, i_batch, i_seq, 1, 
            name, epoch, batch, i_batch, i_seq, 2, 
            name, epoch, batch, i_batch, i_seq, 4
        ))

        os.system(ffmpeg + ' -i ../figs/%s/%04d-%d-%d.png -i ../figs/%s/%04d-%d-%d-%d-%d.png -filter_complex hstack ../figs/%s/%04d-%d-%d-%d-%d.png'%(
            name, epoch, batch, i_batch, 
            name, epoch, batch, i_batch, i_seq, 3, 
            name, epoch, batch, i_batch, i_seq, 5
        ))

        os.system(ffmpeg + ' -i ../figs/%s/%04d-%d-%d-%d-%d.png -i ../figs/%s/%04d-%d-%d-%d-%d.png -filter_complex vstack -y ../figs/%s/%04d-%d-%d-%d.png'%(
            name, epoch, batch, i_batch, i_seq, 4,
            name, epoch, batch, i_batch, i_seq, 5,
            name, epoch, batch, i_batch, i_seq
        ))

        os.remove('../figs/%s/%04d-%d-%d-%d-1.png'%(name, epoch, batch, i_batch, i_seq))
        os.remove('../figs/%s/%04d-%d-%d-%d-2.png'%(name, epoch, batch, i_batch, i_seq))
        os.remove('../figs/%s/%04d-%d-%d-%d-3.png'%(name, epoch, batch, i_batch, i_seq))
        os.remove('../figs/%s/%04d-%d-%d-%d-4.png'%(name, epoch, batch, i_batch, i_seq))
        os.remove('../figs/%s/%04d-%d-%d-%d-5.png'%(name, epoch, batch, i_batch, i_seq))

    # print("#### Generate palette ####")
    # os.system(ffmpeg + ' -i ../figs/%s-%04d-%d-%d-%%d.png -filter_complex "[0:v] palettegen" -y palette.png'%(name, epoch, batch, i_batch))
    # print("#### Generate GIF ####")
    # print(ffmpeg + ' -i ../figs/%s-%04d-%d-%d-%%d.png -i palette.png -filter_complex "[0:v][1:v] paletteuse" -y ../figs/%s-%04d-%d-%d.gif'%(name, epoch, batch, i_batch, name, epoch, batch, i_batch))
    # os.system(ffmpeg + ' -i ../figs/%s-%04d-%d-%d-%%d.png -i palette.png -filter_complex "[0:v][1:v] paletteuse" -loop 1 -y ../figs/%s-%04d-%d-%d.gif'%(name, epoch, batch, i_batch, name, epoch, batch, i_batch))
    # print("#### Remove palette ####")
    # os.remove('palette.png')

    os.remove('../figs/%s/%04d-%d-%d.png'%(name, epoch, batch, i_batch))
    # for i_seq in range(len(syn_z[0]) - 1):
    #     os.remove('../figs/%s-%04d-%d-%d-%d.png'%(name, epoch, batch, i_batch, i_seq))
