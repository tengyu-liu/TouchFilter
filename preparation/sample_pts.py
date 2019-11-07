import os

import numpy as np
import random
import scipy.io as sio
import trimesh as tm
from mayavi import mlab

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

obj_base = '../data/hand'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

mat_data = sio.loadmat('../data/grasp/cup%d/cup%d_grasping_60s_1.mat'%(1, 1))['glove_data']

frame = 0 #np.random.randint(1000)

xpos = mat_data[frame, 1 + 6 * 3 : 1 + 28 * 3].reshape([-1, 3])
xquat = mat_data[frame, 1 + 28 * 3 + 6 * 4 : 1 + 28 * 3 + 28 * 4].reshape([-1, 4])

# for pid in range(21):
#     p = stl_dict[parts[pid]]
#     p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
#     p.apply_translation(xpos[pid,:])

print(sum([int(stl_dict[p].area * 100000) for p in parts]))

for pid in range(21):
    p = stl_dict[parts[pid]]
    # p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
    # p.apply_translation(xpos[pid,:])
    pts, face_idx = p.sample(int(p.area * 100000), return_index=True)
    normals = p.face_normals[face_idx]
    # mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces)
    # mlab.points3d(pts[:,0], pts[:,1], pts[:,2], scale_factor=0.001)
    # mlab.quiver3d(pts[:,0], pts[:,1], pts[:,2], normals[:,0], normals[:,1], normals[:,2])
    # mlab.show()
    np.save('../data/%s.surface_sample.npy'%parts[pid], pts)
    np.save('../data/%s.sample_normal.npy'%parts[pid], normals)

# mlab.show()

# remain = tm.load_mesh(os.path.join(os.path.dirname(__file__), 'hand_removed.obj'))
# remaining_faces = remain.vertices[remain.faces]

# sensor_face_ids = {p:[] for p in parts}

# for p in parts:
#     for fid in range(len(stl_dict[p].faces)):
#         f = stl_dict[p].vertices[stl_dict[p].faces[fid]].reshape([9])
#         if not np.any(np.linalg.norm(np.expand_dims(f, axis=0) - remaining_faces.reshape([-1,9]), axis=-1) < 1e-4):
#             sensor_face_ids[p].append(fid)
#     print(p, len(sensor_face_ids[p]))

# stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

# for p in parts:
#     faces = []
#     face_normals = []
#     for face_id in sensor_face_ids[p]:
#         if p == 'palm' or random.random() < 0.1:
#             faces.append(stl_dict[p].vertices[stl_dict[p].faces[face_id]])
#             face_normals.append(stl_dict[p].face_normals[face_id])

#     faces = np.array(faces)
#     face_normals = np.array(face_normals)
#     print(p, faces.shape, face_normals.shape)
#     np.save('preparation/' + p + '.faces', faces)
#     np.save('preparation/' + p + '.face_normals', face_normals)

# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # ax = plt.subplot(111, projection='3d')
# # faces = np.mean(faces, axis=1)
# # print(faces.shape)
# # ax.scatter(faces[:,0], faces[:,1], faces[:,2], s=1)
# # ax.set_xlim([-0.27, 0.03])
# # ax.set_ylim([-0.4, -0.1])
# # ax.set_zlim([0.13, 0.43])
# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # plt.show()