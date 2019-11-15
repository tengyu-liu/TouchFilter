"""
The purpose of this script is to create a triplet of images for several graspings:
1. Standard grasping hand + obj
2. Sampled points on the hands
3. Hand points with color encoding hand-obj distance
4. Hand points with color encoding hand-obj distance, with shape of an expanded hand
"""

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q
import tensorflow as tf

import mat_trans as mt
from forward_kinematics import ForwardKinematic
from CupModel import CupModel

# Metadata
project_root = os.path.join(os.path.dirname(__file__), '..')
parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

# Load grasping hands
cup_rs = defaultdict(list)
obs_zs = defaultdict(list)
cup_id_list = [1,2,3,4,5,6,7,8,9,10]
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}
for i in cup_id_list:
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
        annotation = cup_annotation['%d_%d'%(i,j)]
        for start_end in annotation:
            start, end = [int(x) for x in start_end.split(':')]
            for frame in range(start, end):
                cup_id = i
                cup_rotation = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotation_matrix
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix
                hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                hand_z = np.concatenate([hand_jrot, hand_grot.reshape([9]), hand_gpos])
                cup_rs[cup_id].append(cup_rotation)
                obs_zs[cup_id].append(hand_z)

cup_rs = {i:np.array(x) for (i,x) in cup_rs.items()}
obs_zs = {i:np.array(x) for (i,x) in obs_zs.items()}

for i_image in range(10):
    os.makedirs('imgs/%d'%i_image, exist_ok=True)

    # Load item
    cup_id = random.sample(cup_id_list, 1)[0]
    frame = random.randint(0, obs_zs[cup_id].shape[0] - 1)
    cup_model = tm.load_mesh(os.path.join(project_root, 'data', 'cups/onepiece/%d.obj'%cup_id))
    cup_r = cup_rs[cup_id][frame]
    obs_z = obs_zs[cup_id][frame]

    print(cup_id, frame)

    tf_cup_model = CupModel(cup_id, 199, os.path.join(project_root, 'data', 'cups', 'models'))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Draw image 1
    if True:
        mlab.clf()

        stl_dict = {obj: tm.load_mesh(os.path.join(project_root, 'data', 'hand/%s.STL'%obj)) for obj in parts}

        # Draw cup
        cvert = np.matmul(cup_r, cup_model.vertices.T).T
        mlab.triangular_mesh(cvert[:,0], cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 1, 0))

        mlab.savefig('imgs/%d/img4.png'%i_image)

        # Draw hand
        jrot = obs_z[:22]
        grot = np.reshape(obs_z[22:31], [3, 3])
        gpos = obs_z[31:]

        grot = mt.quaternion_from_matrix(grot)

        qpos = np.concatenate([gpos, grot, jrot])

        xpos, xquat = ForwardKinematic(qpos)

        for pid in range(4, 25):
            p = stl_dict[parts[pid - 4]]
            try:
                p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
                p.apply_translation(xpos[pid,:])
                mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
            except:
                continue

        mlab.savefig('imgs/%d/img1.png'%i_image)

    # Draw image 2
    if True:
        hand_pts = []
        for pid in range(4, 25):
            pts = np.load(os.path.join(project_root, 'data/%s.sample_points.npy'%parts[pid-4]))
            pts = np.matmul(tm.transformations.quaternion_matrix(xquat[pid,:])[:3,:3], pts.T).T + xpos[pid, :]
            hand_pts.append(pts)
            
        hand_pts = np.concatenate(hand_pts, axis=0)
        pts_min, pts_max = hand_pts.min(axis=0), hand_pts.max(axis=0)
        center = (pts_min + pts_max) / 2

        # grid = np.linspace(-0.15, 0.15, 32)
        # grid = np.stack(np.meshgrid(grid, grid, grid), axis=-1).reshape([-1,3])
        # grid_rot = np.matmul(np.linalg.inv(cup_r), grid.T).T

        # dists = sess.run(tf_cup_model.pred, feed_dict={tf_cup_model.x: grid_rot})[0][:,0]

        ax = plt.subplot(111, projection='3d')
        ax.cla()
        ax.axis('off')
        # ax.scatter(grid[:,0], grid[:,1], grid[:,2], s=3, c=dists)
        ax.scatter(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], s=1, c='black')
        ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
        ax.set_ylim([center[1] - 0.15, center[1] + 0.15])
        ax.set_zlim([center[2] - 0.15, center[2] + 0.15])

        for i in range(40):
            ax.view_init(azim = i * 9)
            plt.savefig('imgs/%d/img2_%d.png'%(i_image, i))
        os.system('ffmpeg4 -i imgs/%d/img2_%%d.png -vf palettegen imgs/palette.png'%i_image)
        try:
            os.remove('imgs/%d/img2.gif'%i_image)
        except:
            pass
        os.system('ffmpeg4 -i imgs/%d/img2_%%d.png -i imgs/palette.png -filter_complex "paletteuse" imgs/%d/img2.gif'%(i_image, i_image))
        for i in range(40):
            os.remove('imgs/%d/img2_%d.png'%(i_image, i))
        os.remove('imgs/palette.png')

    # Draw image 3
    if True: 
        pts = np.matmul(np.linalg.inv(cup_r), hand_pts.T).T
        dists = sess.run(tf_cup_model.pred, feed_dict={tf_cup_model.x: pts})[0][:,0]

        colors = np.zeros(dists.shape)

        colors[dists >= -0.005] = 1.0
        # colors[dists != 1.0] = 0.0

        ax = plt.subplot(111, projection='3d')
        ax.cla()
        ax.axis('off')
        ax.scatter(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], vmin=0, vmax=1, s=1, c=colors, cmap='cool')
        ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
        ax.set_ylim([center[1] - 0.15, center[1] + 0.15])
        ax.set_zlim([center[2] - 0.15, center[2] + 0.15])

        for i in range(40):
            ax.view_init(azim = i * 9)
            plt.savefig('imgs/%d/img3_%d.png'%(i_image, i))
        os.system('ffmpeg4 -i imgs/%d/img3_%%d.png -vf palettegen imgs/palette.png'%(i_image))
        try:
            os.remove('imgs/%d/img3.gif'%i_image)
        except:
            pass
        os.system('ffmpeg4 -i imgs/%d/img3_%%d.png -i imgs/palette.png -filter_complex "paletteuse" imgs/%d/img3.gif'%(i_image, i_image))
        for i in range(40):
            os.remove('imgs/%d/img3_%d.png'%(i_image, i))
        os.remove('imgs/palette.png')

    # Draw image 4, 5, 6
    if True: 
        ax = plt.subplot(111, projection='3d')
        ax.cla()
        ax.axis('off')
        ax.scatter(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], s=1, c='k')
        ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
        ax.set_ylim([center[1] - 0.15, center[1] + 0.15])
        ax.set_zlim([center[2] - 0.15, center[2] + 0.15])

        plt.savefig('imgs/%d/img5.png'%i_image)

        ax = plt.subplot(111, projection='3d')
        ax.cla()
        ax.axis('off')
        ax.scatter(hand_pts[:,0], hand_pts[:,1], hand_pts[:,2], s=1, c=dists, cmap='cool')
        ax.set_xlim([center[0] - 0.15, center[0] + 0.15])
        ax.set_ylim([center[1] - 0.15, center[1] + 0.15])
        ax.set_zlim([center[2] - 0.15, center[2] + 0.15])

        plt.savefig('imgs/%d/img6.png'%i_image)

    # if True:
    #     pts = []
    #     xpos, xquat = ForwardKinematic(np.zeros([31]))

    #     for pid in range(4, 25):
    #         p = np.load(os.path.join(project_root, 'data/%s.sample_points.npy'%parts[pid-4]))
    #         p = np.matmul(Q().rotation_matrix, p.T).T
    #         p += xpos[[pid], :]
    #         pts.append(p)
        
    #     pts = np.vstack(pts)

    #     pick = pts[:,2] <= 0.01
    #     pts = pts[pick]
        
    #     plt.clf()
    #     plt.scatter(pts[:,0], -pts[:,1], c=dists[pick])
    #     plt.axis('off')
    #     plt.savefig('img4.png')
