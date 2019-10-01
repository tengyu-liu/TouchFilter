import os

import numpy as np
import scipy.io as sio
import trimesh as tm
from mayavi import mlab

parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

obj_base = '/home/tengyu/Documents/DeepSDF/MujocoSDF/data/hand/my_model-20190305T235921Z-001/my_model/mesh'
stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, '%s.STL'%obj)) for obj in parts}

mat_data = sio.loadmat('/home/tengyu/Documents/DeepSDF/MujocoSDF/data/grasp/cup%d/cup%d_grasping_60s_1.mat'%(1, 1))['glove_data']

frame = 0 #np.random.randint(1000)

xpos = mat_data[frame, 1 + 6 * 3 : 1 + 28 * 3].reshape([-1, 3])
xquat = mat_data[frame, 1 + 28 * 3 + 6 * 4 : 1 + 28 * 3 + 28 * 4].reshape([-1, 4])

for pid in range(21):
    p = stl_dict[parts[pid]]
    p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
    p.apply_translation(xpos[pid,:])
    mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))

vert_count = 1
verts = []
faces = []
for p in parts:
    verts.append(stl_dict[p].vertices)
    faces.append(stl_dict[p].faces + vert_count)
    vert_count += stl_dict[p].vertices.shape[0]

verts = np.vstack(verts)
faces = np.vstack(faces)

wf = open('hand.obj', 'w')
for v in verts:
    wf.write('v %f %f %f\n'%(v[0], v[1], v[2]))
for f in faces:
    wf.write('f %d %d %d\n'%(f[0], f[1], f[2]))
wf.close()

mlab.show()
