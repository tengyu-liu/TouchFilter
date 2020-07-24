import copy
import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import trimesh as tm
from plotly.subplots import make_subplots
import plotly.io as pio
from pyquaternion.quaternion import Quaternion as Q

from utils import mat_trans as mt
from utils.forward_kinematics import ForwardKinematic

parts = ['palm', 
          'thumb0', 'thumb1', 'thumb2', 'thumb3',
          'index0', 'index1', 'index2', 'index3',
          'middle0', 'middle1', 'middle2', 'middle3',
          'ring0', 'ring1', 'ring2', 'ring3',
          'pinky0', 'pinky1', 'pinky2', 'pinky3']

parts_ = ['palm', 
          'thumb1', 'thumb2', 'thumb3',
          'index1', 'index2', 'index3',
          'middle1', 'middle2', 'middle3',
          'ring1', 'ring2', 'ring3',
          'pinky1', 'pinky2', 'pinky3']

def floatRgb(mag):
    """ Return a tuple of floats between 0 and 1 for R, G, and B. """
    # Normalize to 0-1
    cmax = np.max(mag)
    cmin = np.min(mag)
    x = (mag-cmin)/(cmax-cmin)
    x[np.isnan(x)] = 0.5
    x[np.isinf(x)] = 0.5
    blue  = (np.minimum(np.maximum(4*(0.75-x), 0.), 1.) * 255).astype(np.int32)
    red   = (np.minimum(np.maximum(4*(x-0.25), 0.), 1.) * 255).astype(np.int32)
    green = (np.minimum(np.maximum(4*np.fabs(x-0.5)-1., 0.), 1.) * 255).astype(np.int32)
    return np.stack([red, green, blue], axis=-1)

colors = floatRgb(np.linspace(0, 15, 16))

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

z_ = np.zeros([31])
jrot = np.zeros([22])
gpos = np.zeros([3])
grot = mt.quaternion_from_matrix(np.eye(3))
qpos = np.concatenate([gpos, grot, jrot])
xpos, xquat = ForwardKinematic(qpos)
stl_dict = {obj: tm.load_mesh(os.path.join('/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/data/hand', '%s.STL'%obj)) for obj in parts}
fig_data = []
x, y, z, i, j, k, intensity = [], [], [], [], [], [], []
count = 0
fig_data = []
for pid in range(4, 25):
    if '0' in parts[pid-4]:
      continue
    p = copy.deepcopy(stl_dict[parts[pid - 4]])
    try:
        p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
        p.apply_translation(xpos[pid,:])
        # x.append(self.__zero_stl_dict[self.parts[pid - 4]].vertices[:,0])
        # y.append(self.__zero_stl_dict[self.parts[pid - 4]].vertices[:,1])
        # z.append(self.__zero_stl_dict[self.parts[pid - 4]].vertices[:,2])
        # intensity.append(-np.power(-tm.proximity.signed_distance(cup_model, p.vertices), 1/2))
        # i.append(self.__zero_stl_dict[self.parts[pid - 4]].faces[:,0] + count)
        # j.append(self.__zero_stl_dict[self.parts[pid - 4]].faces[:,1] + count)
        # k.append(self.__zero_stl_dict[self.parts[pid - 4]].faces[:,2] + count)
        # count += len(p.vertices)
        c = colors[parts_.index(parts[pid-4])]
        fig_data.append(go.Mesh3d(x=p.vertices[:,0], y=p.vertices[:,1], z=p.vertices[:,2], \
                                i=p.faces[:,0], j=p.faces[:,1], k=p.faces[:,2], color='rgb(%d,%d,%d)'%(c[0], c[1], c[2])))
    except:
        raise
fig1 = go.Figure(data=fig_data)
fig1.show()
