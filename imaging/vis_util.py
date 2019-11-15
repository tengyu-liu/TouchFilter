import copy
import os

import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import trimesh as tm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pyquaternion.quaternion import Quaternion as Q

import mat_trans as mt
from forward_kinematics import ForwardKinematic


class VisUtil:
    def __init__(self, size=[640, 480]):
        mlab.options.offscreen = True

        self.parts = ['palm', 
                    'thumb0', 'thumb1', 'thumb2', 'thumb3',
                    'index0', 'index1', 'index2', 'index3',
                    'middle0', 'middle1', 'middle2', 'middle3',
                    'ring0', 'ring1', 'ring2', 'ring3',
                    'pinky0', 'pinky1', 'pinky2', 'pinky3']

        xpos, xquat = ForwardKinematic(np.zeros([53]))

        obj_base = os.path.join(os.path.dirname(__file__), '../../data')
        stl_dict = {obj: np.load(os.path.join(obj_base, '%s.sample_points.npy'%obj)) for obj in self.parts}

        start = 0
        end = 0

        self.pts = []

        for pid in range(4, 25):
            if '0' in self.parts[pid - 4]:
                continue
            p = stl_dict[self.parts[pid - 4]]
            p = np.matmul(Q().rotation_matrix, p.T).T
            p += xpos[[pid], :]
            self.pts.append(p)
        
        self.pts = np.vstack(self.pts)

        self.stl_dict = {obj: tm.load_mesh(os.path.join(obj_base, 'hand/%s.STL'%obj)) for obj in self.parts}
        self.cup_models = {i: tm.load_mesh(os.path.join(obj_base, 'cups/onepiece/%d.obj'%i)) for i in range(1,9)}

        self.fig = Figure(figsize=(size[0] / 100., size[1] / 100.), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.width, self.height = self.fig.get_size_inches() * self.fig.get_dpi()
        self.width, self.height = int(self.width), int(self.height)
        mlab.figure(size=size)



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

    def visualize_hand(self, weight):
        img_back = []
        img_front = []

        for i in range(len(weight)):
            if i == 3:
                break
            self.fig.clf()
            ax = self.fig.gca()
            pick = self.pts[:,2] > 0.001
            _ = ax.scatter(self.pts[pick,0], self.pts[pick,1], c=weight[i,pick,0])
            ax.axis('off')
            self.fig.colorbar(_)
            self.canvas.draw()
            image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape([self.height, self.width, 3])
            img_back.append(image)

            self.fig.clf()
            ax = self.fig.gca()
            pick = self.pts[:,2] <= 0.001
            _ = ax.scatter(self.pts[pick,0], self.pts[pick,1], c=weight[i,pick,0])
            ax.axis('off')
            self.fig.colorbar(_)
            self.canvas.draw()
            image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape([self.height, self.width, 3])
            img_front.append(image)

        return np.array(img_back), np.array(img_front)

    def plot_e(self, syn_e, obs_e):
        self.fig.clf()
        ax = self.fig.gca()
        ax.plot(obs_e - syn_e)
        ax.set_yscale('symlog')
        self.canvas.draw()
        image = np.fromstring(self.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape([self.height, self.width, 3])
        return np.expand_dims(image, axis=0)

    def visualize(self, cup_id, cup_r, hand_z):
        imgs = []

        for i in range(len(cup_r)):
            if i == 3:
                break
            mlab.clf()
            cup_model = self.cup_models[cup_id]
            cvert = np.matmul(cup_r[i], cup_model.vertices.T).T
            mlab.triangular_mesh(cvert[:,0], cvert[:,1], cvert[:,2], cup_model.faces, color=(0, 1, 0))

            z_ = hand_z[i]
            jrot = z_[:22]
            grot = np.reshape(z_[22:28], [3, 2])
            gpos = z_[28:]

            grot = mt.quaternion_from_matrix(self.rotation_matrix(grot))

            qpos = np.concatenate([gpos, grot, jrot])

            xpos, xquat = ForwardKinematic(qpos)

            for pid in range(4, 25):
                p = copy.deepcopy(self.stl_dict[self.parts[pid - 4]])
                try:
                    p.apply_transform(tm.transformations.quaternion_matrix(xquat[pid,:]))
                    p.apply_translation(xpos[pid,:])
                    mlab.triangular_mesh(p.vertices[:,0], p.vertices[:,1], p.vertices[:,2], p.faces, color=(1, 0, 0))
                except:
                    continue
            imgs.append(mlab.screenshot())

        return np.array((imgs))

if __name__ == "__main__":
    vu = VisUtil()
    z = np.zeros([53])
    z[1:44:2] = 1.0
    z[44] = 1
    z[47] = 1
    z = np.array([z,z,z,z])
    img = vu.visualize(1, np.array([np.eye(3)]*4), z)
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(img[i])
    plt.show()
