import os
import sys
import pickle

import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import trimesh as tm

sys.path.append(os.path.dirname(__file__))

import mat_trans as mt
from forward_kinematics import ForwardKinematic

pca_components = np.load('data/pca_components.npy')
pca_mean = np.load('data/pca_mean.npy')
pca_var = np.load('data/pca_variance.npy')

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

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
    cup_model = tm.load_mesh('data/cups/onepiece/%d.obj'%cup_id)
    cvert = np.matmul(cup_r, cup_model.vertices.T).T
    if offset == 0:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 1, 0))
    else:
        mlab.triangular_mesh(cvert[:,0] + offset, cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 0, 1))

    z_ = np.matmul(hand_z, np.expand_dims(np.sqrt(pca_var), axis=-1) * pca_components) + pca_mean
    jrot = np.reshape(z_[:44], [22, 2])
    grot = np.reshape(z_[44:50], [3, 2])
    gpos = z_[50:]

    jrot = np.arcsin((jrot / np.linalg.norm(jrot, axis=-1, keepdims=True))[:,0])
    grot = mt.quaternion_from_matrix(rotation_matrix(grot))

    qpos = np.concatenate([gpos, grot, jrot])

    xpos, xquat = ForwardKinematic(qpos)

    obj_base = '/home/tengyu/Documents/DeepSDF/MujocoSDF/data/hand/my_model-20190305T235921Z-001/my_model/mesh'
    stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

    for pid in range(4, 25):
        p = stl_dict[parts[pid - 4]]
        p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
        p.apply_translation(xpos[pid,:])
        mlab.triangular_mesh(p.vertices[:,0] + offset, p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))


if __name__ == '__main__':
    name = sys.argv[1]
    epoch = int(sys.argv[2])
    batch = int(sys.argv[3])

    data = pickle.load(open('figs/%s/%04d-%d.pkl'%(name, epoch, batch), 'rb'))

    cup_id = data['cup_id']
    cup_r = data['cup_r']
    obs_z = data['obs_z']
    ini_z = data['ini_z']
    syn_z = data['syn_z']

    for i in range(len(cup_r)):
        visualize(cup_id, cup_r[i], obs_z[i], offset=0)
        visualize(cup_id, cup_r[i], ini_z[i], offset=1)
        visualize(cup_id, cup_r[i], syn_z[i], offset=2)
        print(np.linalg.norm(ini_z[i] - syn_z[i]))
        mlab.show()

        plt.plot(np.arange(36), obs_z[i], c='green')
        plt.plot(np.arange(36), ini_z[i], c='red')
        plt.plot(np.arange(36), syn_z[i], c='blue')
        plt.show()