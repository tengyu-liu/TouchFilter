from itertools import permutations
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import trimesh
from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues
from manopth.rot6d import robust_compute_rotation_matrix_from_ortho6d

from mesh_util import load_faces
# np.random.seed(0)
# torch.manual_seed(0)


class HandModel:
  def __init__(
    self, 
    n_handcode=6, root_rot_mode='ortho6d', robust_rot=False, flat_hand_mean=True,
    mano_path='third_party/manopth/mano/models', n_contact=3):
    self.n_contact = n_contact
    self.device = torch.device('cuda')
    device = self.device
    self.contact_permutations = [list(p) for p in permutations(np.arange(self.n_contact))]
    if n_handcode == 45:
      self.layer = ManoLayer(root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, use_pca=False).to(device)
    else:
      self.layer = ManoLayer(ncomps=n_handcode, root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, flat_hand_mean=flat_hand_mean).to(device)
    self.code_length = n_handcode + 6
    if root_rot_mode != 'axisang':
      self.code_length += 3

    self.forward_base_ids = [34,268]
    self.sideway_base_ids = [96,115]
    self.facedir_base_ids = [218,95]

    self.texture_coords = torch.tensor(self.get_texture_coords().reshape([1, 1, -1, 2]) * 2 - 1).float().to(device)
    self.original_faces = self.layer.th_faces.detach().cpu().numpy()
    self.faces = load_faces('mano_no_thumb.obj')
    self.keep_verts = list(set(self.faces.reshape([-1])))

    self.faces_in_new_verts = np.array([[self.keep_verts.index(x) for x in _f] for _f in self.faces])

    self.num_points = len(self.keep_verts)
    self.verts_eye = torch.tensor(np.eye(self.num_points)).float().to(device)
    self.n1_mat = self.verts_eye[self.faces_in_new_verts[:,0]]   # F x V
    self.n2_mat = self.verts_eye[self.faces_in_new_verts[:,1]]   # F x V
    self.n3_mat = self.verts_eye[self.faces_in_new_verts[:,2]]   # F x V
    self.fv_total = self.n1_mat.sum(0) + self.n2_mat.sum(0) + self.n3_mat.sum(0) # V
    self.neighbors = defaultdict(set)
    for v1,v2,v3 in self.faces:
      self.neighbors[v1].add(v1)
      self.neighbors[v1].add(v2)
      self.neighbors[v1].add(v3)
      self.neighbors[v2].add(v1)
      self.neighbors[v2].add(v2)
      self.neighbors[v2].add(v3)
      self.neighbors[v3].add(v1)
      self.neighbors[v3].add(v2)
      self.neighbors[v3].add(v3)
    self.neighbors = np.array([list(self.neighbors[i]) for i in range(self.num_points)])

    if os.path.exists('mano_no_thumb_manifold_distances.pkl'):
      self.mano_manifold_distances = pickle.load(open('mano_no_thumb_manifold_distances.pkl', 'rb'))
    else:
      self.mano_manifold_distances = self.compute_manifold_distances()
      pickle.dump(self.mano_manifold_distances, open('mano_no_thumb_manifold_distances.pkl', 'wb'))

  def compute_manifold_distances(self):
    distances = np.zeros([self.num_points, self.num_points])
    zero_verts = self.get_vertices(torch.normal(0,1,size=[1,15], device=self.device) * 0.001)[0].detach().cpu().numpy()
    for i,js in enumerate(self.neighbors):
      for j in js:
        if i > j:
          d = np.linalg.norm(zero_verts[i] - zero_verts[j])
          distances[i][j] = d
          distances[j][i] = d
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import floyd_warshall
    graph = csr_matrix(distances)
    dist_matrix = floyd_warshall(csgraph=graph, directed=False, return_predecessors=False)
    return torch.tensor(dist_matrix).float().to(self.device)

  def load_obj(self, path):
    f = open(path)
    vertices = []
    faces = []
    temp_texture_coordinates = []
    lines = f.readlines()
    
    for line in lines:
      line = line.strip().split(' ')
      if line[0] == 'v':
        vertices.append(list(map(float, line[1:])))
      if line[0] == 'vt':
        temp_texture_coordinates.append(list(map(float, line[1:])))
      
    temp_texture_coordinates = np.array(temp_texture_coordinates)
    texture_coordinates = np.zeros([len(vertices), 2])
    for line in lines:
      line = line.strip().split(' ')
      if line[0] == 'f':
        v1,vt1,_ = list(map(int, line[1].split('/')))
        v2,vt2,_ = list(map(int, line[2].split('/')))
        v3,vt3,_ = list(map(int, line[3].split('/')))
        if np.abs(texture_coordinates[v1-1]).sum() == 0:
          texture_coordinates[v1-1] = temp_texture_coordinates[vt1-1]
        if np.abs(texture_coordinates[v2-1]).sum() == 0:
          texture_coordinates[v2-1] = temp_texture_coordinates[vt2-1]
        if np.abs(texture_coordinates[v3-1]).sum() == 0:
          texture_coordinates[v3-1] = temp_texture_coordinates[vt3-1]
    return np.array(vertices), texture_coordinates

  def get_texture_coords(self):
    old_mano_verts = trimesh.load('mano.obj', process=False).vertices
    texture_verts, texture_coords = self.load_obj('mano_with_uv_full.obj')
    old_mano_texture = np.zeros_like(texture_coords)
    distances = np.linalg.norm(np.expand_dims(old_mano_verts, axis=1) - np.expand_dims(texture_verts, axis=0), axis=-1)
    mappings = np.argmin(distances, axis=1)
    return texture_coords[mappings]

  def test_texture(self):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from skimage.io import imread
    texture = imread('mano.png') / 255.
    w = texture.shape[0]
    h = texture.shape[1]
    v = trimesh.load('mano.obj', process=False).vertices
    c = self.texture_coords
    c[:,0] *= w
    c[:,1] *= h
    c = c.astype(int)
    ax = plt.subplot(111, projection='3d')
    for p in range(len(v)):
      ax.scatter(v[p,0], v[p,1], v[p,2], c=texture[c[p,0], c[p,1]])
    ax.axis('off')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.show()

  def get_vertices(self, hand_code):
    hand_trans = hand_code[:,:3]
    hand_theta = hand_code[:,3:]
    return self.layer(hand_theta)[0][:,self.keep_verts,:] / 120 + hand_trans.unsqueeze(1)

  def get_all_vertices(self, hand_code):
    hand_trans = hand_code[:,:3]
    hand_theta = hand_code[:,3:]
    return self.layer(hand_theta)[0] / 120 + hand_trans.unsqueeze(1)

  def get_joints(self, hand_code):
    hand_trans = hand_code[:,:3]
    hand_theta = hand_code[:,3:]
    return self.layer(hand_theta)[1] / 120 + hand_trans.unsqueeze(1)

  def closest_point(self, hand_code, x):
    # x: B x K x 3
    B = x.shape[0]
    K = x.shape[1]
    hand_verts = self.get_vertices(hand_code) # B x V x 3
    distances = torch.norm(hand_verts.unsqueeze(2) - x.unsqueeze(1), dim=-1) # B x V x K
    argmin_dist = torch.argmin(distances, 1).detach()  # B x K
    closest_points = []
    for k in range(K):
      closest_points.append(hand_verts[torch.arange(B), argmin_dist[:,k]])  # [B x 3] x K
    closest_points = torch.stack(closest_points, 1)
    return closest_points

  def back_direction(self, z=None, verts=None):
    if verts is None:
      if z is None:
        raise ValueError
      verts = self.get_vertices(z)
    vector = verts[:,self.facedir_base_ids[0],:] - verts[:,self.facedir_base_ids[1],:]
    return vector / torch.norm(vector, dim=-1, keepdim=True)

  def forward_direction(self, z=None, verts=None):
    if verts is None:
      if z is None:
        raise ValueError
      verts = self.get_vertices(z)
    vector = verts[:,self.forward_base_ids[1],:] - verts[:,self.forward_base_ids[0],:]
    return vector / torch.norm(vector, dim=-1, keepdim=True)

  def side_direction(self, z=None, verts=None):
    if verts is None:
      if z is None:
        raise ValueError
      verts = self.get_vertices(z)
    vector = verts[:,self.sideway_base_ids[1],:] - verts[:,self.sideway_base_ids[0],:]
    return vector / torch.norm(vector, dim=-1, keepdim=True)

  def flip(self, z, axis):
    # z: B x 15
    # axis: B x 3
    rot = robust_compute_rotation_matrix_from_ortho6d(z[:,3:9])
    rot2 = batch_rodrigues(axis / torch.norm(axis, dim=-1, keepdim=True) * np.pi).view([-1,3,3])
    trans = z[:,:3]
    param = z[:,9:]
    return torch.cat([trans, torch.matmul(rot2, rot)[:,:,:2].transpose(1,2).reshape([-1,6]), param], dim=-1)

  def align(self, z, a, b):
    # z: B x 15
    # a: B x 3
    # b: B x 3
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    B = a.shape[0]
    axis = torch.cross(a, b)
    cosA = (a * b).sum(1)
    k = 1 / (1 + cosA)
    ax = axis[:,0]
    ay = axis[:,1]
    az = axis[:,2]
    rot2 = torch.stack([ax*ax*k + cosA, ay*ax*k - az, az*ax*k + ay, ax*ay*k + az, ay*ay*k + cosA, az*ay*k - ax, ax*az*k - ay, ay*az*k + ax, az*az*k + cosA]).view([B,3,3])
    rot = robust_compute_rotation_matrix_from_ortho6d(z[:,3:9])
    trans = z[:,:3]
    param = z[:,9:]
    return torch.cat([trans, torch.matmul(rot2, rot)[:,:,:2].transpose(1,2).reshape([-1,6]), param], dim=-1)


  def texture_color_per_vertex(self, texture):
    B = texture.shape[0]
    return torch.nn.functional.grid_sample(texture, self.texture_coords.repeat(B,1,1,1)).squeeze().transpose(1,2)  # output: B x V x C

  def get_surface_normals(self, z=None):
    verts = self.get_all_vertices(z)
    
    B = verts.shape[0]
    V = verts.shape[1]
    
    # get all face verts
    fv1 = verts[:,self.faces[:,0],:]
    fv2 = verts[:,self.faces[:,1],:]
    fv3 = verts[:,self.faces[:,2],:]

    # compute normals
    vn1 = torch.cross((fv1-fv3), (fv2-fv1))   # B x F x 3
    vn2 = torch.cross((fv2-fv1), (fv3-fv2))   # B x F x 3
    vn3 = torch.cross((fv3-fv2), (fv1-fv3))   # B x F x 3

    assert (np.all(vn1.detach().cpu().numpy()!=0))
    assert (np.all(vn2.detach().cpu().numpy()!=0))
    assert (np.all(vn3.detach().cpu().numpy()!=0))

    vn1 = vn1 / torch.norm(vn1, dim=-1, keepdim=True)
    vn2 = vn2 / torch.norm(vn2, dim=-1, keepdim=True)
    vn3 = vn3 / torch.norm(vn3, dim=-1, keepdim=True)

    # aggregate normals
    normals = (torch.einsum('bfn,fv->bvn', vn1, self.n1_mat) + torch.einsum('bfn,fv->bvn', vn2, self.n2_mat) + torch.einsum('bfn,fv->bvn', vn3, self.n3_mat)) / self.fv_total.unsqueeze(0).unsqueeze(-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    return normals
  
  def manifold_distance(self, ptsA, ptsB):
    # ptsA: B x 3
    # ptsB: B x 3
    # return: shortest distance on graph between ptsA and ptsB
    distances = torch.stack([self._manifold_distance(ptsA, ptsB[:, perm]) for perm in self.contact_permutations], dim=-1).min(-1)[0]
    return distances
  
  def _manifold_distance(self, ptsA, ptsB):
    # TODO: Run optimization on antelope
    return self.mano_manifold_distances[ptsA, ptsB].sum(-1)

if __name__ == "__main__":
  import numpy as np
  import random
  import plotly
  import plotly.graph_objects as go

  hand_model = HandModel()
  z = torch.normal(0,1,size=[1,15]).float().to(hand_model.device) * 1e-6
  verts = hand_model.get_vertices(z)[0].detach().cpu().numpy()
  all_verts = hand_model.get_all_vertices(z)[0].detach().cpu().numpy()
  fig = plotly.tools.make_subplots(1, 1, specs=[[{'type':'surface'}]])

  fig.append_trace(go.Mesh3d(
    x=verts[:,0], y=verts[:,1], z=verts[:,2], 
    i=hand_model.faces_in_new_verts[:,0], j=hand_model.faces_in_new_verts[:,1], k=hand_model.faces_in_new_verts[:,2]
    ), 1, 1)

  normals = hand_model.get_surface_normals(z=z)[0].detach().cpu().numpy()
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  print(normals.shape)
  fig.append_trace(go.Cone(x=verts[:,0], y=verts[:,1], z=verts[:,2], u=normals[:,0], v=normals[:,1], w=normals[:,2], sizemode='absolute', sizeref=1), 1, 1)

  fig.show()
