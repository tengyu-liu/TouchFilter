import os
import numpy as np
from pyquaternion.quaternion import Quaternion as Q
import scipy.io as sio
import pickle
import trimesh as tm
from collections import defaultdict

class DataLoader:
  def __init__(self, flags, data_dir='../data', obj_list=[3], debug=False):
    print('[loading data]')
    self.data_dir = data_dir
    self.obj_list = obj_list
    self.flags = flags

    # load data
    self.obj_pts = {obj_id: tm.load_mesh(os.path.join(data_dir, 'cups/onepiece/%d.obj'%obj_id)).sample(20000) for obj_id in obj_list}

    cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(data_dir, 'cup_video_annotation.txt')).readlines()}

    obs_zs = defaultdict(list)
    palm_directions = defaultdict(list)

    all_zs = []

    for i in self.obj_list:
        for j in range(1,11):
            mat_data = sio.loadmat(os.path.join(data_dir, 'grasp/cup%d/cup%d_grasping_60s_%d.mat'%(i,i,j)))['glove_data']
            annotation = cup_annotation['%d_%d'%(i,j)]
            for frame in range(len(mat_data)):
                cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                hand_grot = Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4]).rotation_matrix[:,:2]
                hand_gpos = mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation
                hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                all_zs.append(hand_z)
            for start_end in annotation:
                start, end = [int(x) for x in start_end.split(':')]
                for frame in range(start, end):
                    cup_id = i
                    cup_translation = mat_data[frame, 1 + 27 * 3 : 1 + 28 * 3]
                    hand_jrot = mat_data[frame, 1 + 28 * 7 + 7 : 1 + 28 * 7 + 29]
                    hand_grot = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotation_matrix[:,:2]
                    hand_gpos = Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse.rotate(mat_data[frame, 1 + 1 * 3 : 1 + 2 * 3] - cup_translation)
                    hand_z = np.concatenate([hand_jrot, hand_grot.reshape([6]), hand_gpos])
                    palm_v = (Q(mat_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).inverse * Q(mat_data[frame, 1 + 28 * 3 + 1 * 4 : 1 + 28 * 3 + 2 * 4])).rotate([0,0,1])
                    palm_directions[cup_id].append(palm_v)
                    obs_zs[cup_id].append(hand_z)

    self.obs_zs = {i : np.array(x) for (i,x) in obs_zs.items()}
    self.palm_directions = {i : np.array(x) for (i,x) in palm_directions.items()}
    all_zs = np.array(all_zs)

    self.z_stddev = np.std(all_zs, axis=0, keepdims=True)
    self.z_mean = np.mean(all_zs, axis=0, keepdims=True)
    self.z_min = np.tile(np.min(all_zs, axis=0, keepdims=True), [self.flags.batch_size, 1])
    self.z_max = np.tile(np.max(all_zs, axis=0, keepdims=True), [self.flags.batch_size, 1])

    self.min_data_size = min(len(x) for x in self.obs_zs.values())
    self.obs_z2s = {i: np.random.normal(0, 1, [self.obs_zs[i].shape[0], self.flags.n_latent_factor]) for i in self.obs_zs.keys()}


  def fetch(self):
    batch_idx = {i: np.random.permutation(len(self.obs_zs[i])) for i in self.obs_zs}
    
    for curr_iter in range(int(self.min_data_size * len(self.obj_list) // self.flags.batch_size)):
      obj_id = self.obj_list[curr_iter % len(self.obj_list)]
      item_id = int(curr_iter // len(self.obj_list))
      idx = batch_idx[obj_id][item_id * self.flags.batch_size : (item_id + 1) * self.flags.batch_size]
      obs_z = self.obs_zs[obj_id][idx]
      obs_obj = self.sample_pts(obj_id, self.flags.n_obj_pts)
      obs_z2 = self.obs_z2s[obj_id][idx]
      yield obj_id, item_id, obs_z, obs_z2, obs_obj, idx

  def update_z(self, obj_id, z, idx):
      self.obs_z2s[obj_id][idx] = z

  def sample_pts(self, obj_id, n_pts):
    rand_id = np.random.randint(0, len(self.obj_pts[obj_id])-1, size=(self.flags.batch_size, n_pts)) 
    return self.obj_pts[obj_id][rand_id]

  def restore(self, path):
    self.obs_z2s = pickle.load(open(path, 'rb'))[-1]