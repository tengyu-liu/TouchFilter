import copy
import os
import time
from collections import defaultdict

import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio
import trimesh as tm
from pyquaternion.quaternion import Quaternion as Q

import mat_trans as mt
from forward_kinematics import ForwardKinematic

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

cup_models = {i: tm.load_mesh('../../data/cups/onepiece/%d.obj'%i) for i in range(1,9)}
obj_base = '../../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

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

def visualize(cup_id, cup_r, hand_z, offset=0):
    cup_model = cup_models[cup_id]
    cvert = np.matmul(cup_r, cup_model.vertices.T).T
    if offset == 0:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 1, 0))
    else:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 0, 1))

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
            mlab.triangular_mesh(p.vertices[:,0] + offset, p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
        except:
            continue

project_root = os.path.join(os.path.dirname(__file__), '../..')

# load obj
cup_id_list = [1,2,3,5,6,7,8]

cup_models = {cup_id: tm.load_mesh(os.path.join(project_root, 'data/cups/onepiece/%d.obj'%cup_id)) for cup_id in cup_id_list}
print('Cup models loaded.')

# load data
cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(project_root, 'data/cup_video_annotation.txt')).readlines()}

cup_rs = defaultdict(list)
obs_zs = defaultdict(list)
frames = defaultdict(list)
js = defaultdict(list)

total = 0
for i in cup_id_list:
    for j in range(1,11):
        mat_data = sio.loadmat(os.path.join(project_root, 'data/grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
        annotation = cup_annotation['%d_%d'%(i,j)]
        # for frame in range(len(mat_data)):
        #     cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
        #     hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
        #     hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
        #     hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
        #     hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
        #     all_zs.append(hand_z)
        for start_end in annotation:
            start, end = [int(x) for x in start_end.split(':')]
            frame = int((start + end) / 2)
            cup_id = i
            # cup_rotation = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotation_matrix
            cup_rotation = Q().rotation_matrix
            cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
            hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
            hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
            hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
            hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
            cup_rs[cup_id].append(cup_rotation)
            obs_zs[cup_id].append(hand_z)
            frames[cup_id].append([start, end])
            js[cup_id].append(j)
            total += 1

for cup_id in cup_id_list:
    i = 0
    for cup_r, obs_z, j, frame in zip(cup_rs[cup_id], obs_zs[cup_id], js[cup_id], frames[cup_id]):
        t0 = time.time()
        mlab.clf()
        visualize(cup_id, cup_r, obs_z)
        mlab.show()
        # i += 1
        # t1 = time.time()
        # if i < total:
        #     print('\r%d/%d ETA: %f sec'%(i, total, (t1 - t0) * (total - i)), end='', flush=True)
        # else:
        #     print('\nDone.')

"""
cup_id record start end type
3 1 89 127 wide pinch index
3 1 161 235 power grip thumb in
3 1 269 319 power grip thumb in
3 3 439 527 power grip thumb out
3 4 69 157 tri-tip pinch
3 4 223 291 wide pinch (?)
3 4 461 521 wide pinch (?)
3 4 563 649 precision thumb 4-finger
3 4 733 831 precision thumb 4-finger
3 5 215 317 soft clip
3 5 387 439 soft clip
3 5 511 561 tri-tip pinch
3 5 619 691 precision thumb index (?)
3 5 877 945 precision circular
3 6 73 125 precision circular
3 6 179 287 power grip thumb in
3 6 333 417 power grip thumb out (?)
3 6 971 997 precision thumb 4-finger
3 7 70 119 precision circular
3 7 529 603 precision circular
3 7 637 709 precision thumb 4-finger
3 7 735 795 wide pinch middle
3 7 927 997 small sphere palm
3 10 91 151 tri-tip pinch
3 10 215 307 precision thumb 2-finger

* Not inspected with 3D models. Only with 2D single view rendering from 3D models.
* Conclusion: Use cup #3 for most diverse graspings. 
"""