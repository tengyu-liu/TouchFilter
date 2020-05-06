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

from . import mat_trans as mt
from .forward_kinematics import ForwardKinematic

# if os.name != 'nt':
#     pio.orca.config.use_xvfb = True

class Visualizer:
    def __init__(self):
        self.parts = ['palm', 
                    'thumb0', 'thumb1', 'thumb2', 'thumb3',
                    'index0', 'index1', 'index2', 'index3',
                    'middle0', 'middle1', 'middle2', 'middle3',
                    'ring0', 'ring1', 'ring2', 'ring3',
                    'pinky0', 'pinky1', 'pinky2', 'pinky3']

        self.obj_base = os.path.join(os.path.dirname(__file__), '../../data/hand')
        __zero_jrot = np.zeros([22]) + 0.3
        __zero_grot = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        __zero_gpos = np.zeros([3])
        __zero_grot = mt.quaternion_from_matrix(self.rotation_matrix(__zero_grot))
        __zero_qpos = np.concatenate([__zero_gpos, __zero_grot, __zero_jrot])
        __zero_xpos, __zero_xquat = ForwardKinematic(__zero_qpos)
        self.__zero_stl_dict = {obj: tm.load_mesh(os.path.join(self.obj_base, '%s.STL'%obj)) for obj in self.parts}
        pts = {obj: np.load(os.path.join(self.obj_base, '..', '%s.sample_points.npy'%obj)) for obj in self.parts if '0' not in obj}
        self.__zero_pts = []

        for pid in range(4, 25):
            p = copy.deepcopy(self.__zero_stl_dict[self.parts[pid - 4]])
            try:
                p.apply_transform(tm.transformations.quaternion_matrix(__zero_xquat[pid,:]))
                p.apply_translation(__zero_xpos[pid,:])
                self.__zero_stl_dict[self.parts[pid - 4]] = p
                obj = self.parts[pid - 4]
                if '0' not in obj:
                    self.__zero_pts.append(np.matmul(mt.quaternion_matrix(__zero_xquat[pid,:])[:3,:3], pts[obj].T).T + __zero_xpos[pid,:])
            except:
                continue
        self.__zero_pts = np.vstack(self.__zero_pts)

    def rotation_matrix(self, rot):
        a1 = rot[:,0]
        a2 = rot[:,1]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        eye = np.eye(4)
        eye[:3,:3] = np.stack([b1, b2, b3], axis=-1)
        return eye

    def visualize_distance(self, cup_id, hand_z, save_path):
        cup_model = tm.load_mesh(os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/3.obj'))
        z_ = hand_z
        jrot = z_[:22]
        grot = np.reshape(z_[22:28], [3, 2])
        gpos = z_[28:]
        grot = mt.quaternion_from_matrix(self.rotation_matrix(grot))
        qpos = np.concatenate([gpos, grot, jrot])
        xpos, xquat = ForwardKinematic(qpos)
        stl_dict = {obj: tm.load_mesh(os.path.join(self.obj_base, '%s.STL'%obj)) for obj in self.parts}
        fig_data = []
        x, y, z, i, j, k, intensity = [], [], [], [], [], [], []
        count = 0
        fig_data = [go.Mesh3d(x=cup_model.vertices[:,0], y=cup_model.vertices[:,1], z=cup_model.vertices[:,2], \
                                i=cup_model.faces[:,0], j=cup_model.faces[:,1], k=cup_model.faces[:,2], color='lightpink')]
        for pid in range(4, 25):
            p = copy.deepcopy(stl_dict[self.parts[pid - 4]])
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
                fig_data.append(go.Mesh3d(x=p.vertices[:,0], y=p.vertices[:,1], z=p.vertices[:,2], \
                                        i=p.faces[:,0], j=p.faces[:,1], k=p.faces[:,2], color='lightblue'))
            except:
                raise
        # Draw figure 1
        fig1 = go.Figure(data=fig_data)
        camera = dict(eye=dict(x=1, y=1, z=1))
        fig1.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
        fig1.write_image('%s-grasp.png'%save_path)
        # Draw figure 2
        # x, y, z, i, j, k, intensity = map(np.hstack, [x, y, z, i, j, k, intensity])
        # fig2 = go.Figure(data=go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, intensity=intensity, showscale=False))
        # camera = dict(eye=dict(x=0, y=0, z=-2), up=dict(x=0, y=-1, z=0))
        # fig2.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
        # fig2.write_image('%s-dist.png'%save_path)

    def visualize_weight(self, hand_z, hand_w, save_path=None, ax=None, c='white'):
        cup_model = tm.load_mesh(os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/3.obj'))
        z_ = hand_z
        jrot = z_[:22]
        grot = np.reshape(z_[22:28], [3, 2])
        gpos = z_[28:]
        grot = mt.quaternion_from_matrix(self.rotation_matrix(grot))
        qpos = np.concatenate([gpos, grot, jrot])
        xpos, xquat = ForwardKinematic(qpos)
        stl_dict = {obj: tm.load_mesh(os.path.join(self.obj_base, '%s.STL'%obj)) for obj in self.parts}
        fig_data = []
        fig_data = [go.Mesh3d(x=cup_model.vertices[:,0], y=cup_model.vertices[:,1], z=cup_model.vertices[:,2], \
                                i=cup_model.faces[:,0], j=cup_model.faces[:,1], k=cup_model.faces[:,2], color='lightpink')]

        for pid in range(4, 25):
            p = copy.deepcopy(stl_dict[self.parts[pid - 4]])
            try:
                p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
                p.apply_translation(xpos[pid,:])
                fig_data.append(go.Mesh3d(x=p.vertices[:,0], y=p.vertices[:,1], z=p.vertices[:,2], \
                                        i=p.faces[:,0], j=p.faces[:,1], k=p.faces[:,2], color='lightblue'))
            except:
                raise


        fig1 = go.Figure(data=fig_data)
        camera = dict(eye=dict(x=0, y=0, z=-2), up=dict(x=0, y=-1, z=0))
        fig1.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))

        hand_w -= np.min(hand_w)
        hand_w /= np.max(hand_w)
        print(hand_w)

        fig2 = go.Figure(data=[go.Scatter3d(
            x=self.__zero_pts[:,0], 
            y=self.__zero_pts[:,1], 
            z=self.__zero_pts[:,2], 
            mode='markers',
            marker=dict(
                color=hand_w, 
                colorscale='Viridis'
            ))])
        camera = dict(eye=dict(x=0, y=0, z=-2), up=dict(x=0, y=-1, z=0))
        fig2.update_layout(scene_camera=camera, scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0,r=0,t=0,b=0))
        if save_path is None:
            fig1.show()
            fig2.show()
        else:
            fig1.write_image('%s-grasp.png'%save_path)
            fig2.write_image('%s-weight.png'%save_path)
        # if ax is not None:
        #     ax.scatter(-self.__zero_pts[:,2], self.__zero_pts[:,0], -self.__zero_pts[:,1], c=hand_w, s=3)
        #     ax.view_init(elev=0, azim=0)
        #     ax.set_xlim(-0.1,0.1)
        #     ax.set_ylim(-0.05, 0.15)
        #     ax.set_zlim(0.05, 0.25)
        #     ax.plot([-0.1,-0.1,-0.1,-0.1,-0.1], [-0.05,-0.05,0.15,0.15,-0.05], [0.05,0.25,0.25,0.05,0.05], c=c)
        #     ax.axis('off')

if __name__ == '__main__':
    v = Visualizer()
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    w = np.load('../z2_analysis/w.npy')
    for i_batch in range(w.shape[0]):
        for i_z2 in range(w.shape[1]):
            for i_val in range(w.shape[2]):
                print(i_batch, i_z2, i_val)
                ax = plt.subplot(111, projection='3d')
                v.visualize_weight(w[i_batch, i_z2, i_val, :, 0], ax=ax)
                plt.show()