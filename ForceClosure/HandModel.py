from manopth.manolayer import ManoLayer
from manopth.rot6d import robust_compute_rotation_matrix_from_ortho6d
import torch
import numpy as np
import trimesh

class HandModel:
  def __init__(
    self, 
    n_handcode=6, root_rot_mode='ortho6d', robust_rot=False, flat_hand_mean=True,
    mano_path='/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/ForceClosure/third_party/manopth/mano/models'):
    if n_handcode == 45:
      self.layer = ManoLayer(root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, use_pca=False).cuda()
    else:
      self.layer = ManoLayer(ncomps=n_handcode, root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, flat_hand_mean=flat_hand_mean).cuda()
    self.code_length = n_handcode + 6
    if root_rot_mode != 'axisang':
      self.code_length += 3
    self.back_vector = torch.tensor(np.array([0,1,0]).reshape([1, 3, 1])).float().cuda()
    self.palm_vector = torch.tensor(np.array([0,0,1]).reshape([1, 3, 1])).float().cuda()
    self.side_vector = torch.tensor(np.array([1,0,0]).reshape([1, 3, 1])).float().cuda()
    self.texture_coords = torch.tensor(self.get_texture_coords().reshape([1, 1, -1, 2]) * 2 - 1).float().cuda()
    self.faces = self.layer.th_faces.detach().cpu().numpy()
    self.num_points = self.texture_coords.shape[2]
    self.verts_eye = torch.tensor(np.eye(self.num_points)).float().cuda()
    self.n1_mat = self.verts_eye[self.faces[:,0]]   # F x V
    self.n2_mat = self.verts_eye[self.faces[:,1]]   # F x V
    self.n3_mat = self.verts_eye[self.faces[:,2]]   # F x V
    self.fv_total = self.n1_mat.sum(0) + self.n2_mat.sum(0) + self.n3_mat.sum(0) # V

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
    return self.layer(hand_theta, th_trans=hand_trans)[0] / 120
  
  def get_joints(self, hand_code):
    hand_trans = hand_code[:,:3]
    hand_theta = hand_code[:,3:]
    return self.layer(hand_theta, th_trans=hand_trans)[1] / 120

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

  def back_direction(self, hand_code):
    B = hand_code.shape[0]
    d = self.back_vector.repeat([B,1,1])
    rot = robust_compute_rotation_matrix_from_ortho6d(hand_code[:, 3:9])
    d = torch.bmm(rot, d).squeeze(2)
    return d

  def palm_direction(self, hand_code):
    B = hand_code.shape[0]
    d = self.palm_vector.repeat([B,1,1])
    rot = robust_compute_rotation_matrix_from_ortho6d(hand_code[:, 3:9])
    d = torch.bmm(rot, d).squeeze(2)
    return d

  def side_direction(self, hand_code):
    B = hand_code.shape[0]
    d = self.side_vector.repeat([B,1,1])
    rot = robust_compute_rotation_matrix_from_ortho6d(hand_code[:, 3:9])
    d = torch.bmm(rot, d).squeeze(2)
    return d

  def texture_color_per_vertex(self, texture):
    B = texture.shape[0]
    return torch.nn.functional.grid_sample(texture, self.texture_coords.repeat(B,1,1,1)).squeeze().transpose(1,2)  # output: B x V x C

  def get_surface_normals(self, z=None, verts=None):
    if verts is None:
      if z is None:
        raise ValueError
      else:
        verts = self.get_vertices(z)
    
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

    vn1 = vn1 / torch.norm(vn1, dim=-1, keepdim=True)
    vn2 = vn2 / torch.norm(vn2, dim=-1, keepdim=True)
    vn3 = vn3 / torch.norm(vn3, dim=-1, keepdim=True)

    # aggregate normals
    normals = (torch.einsum('bfn,fv->bvn', vn1, self.n1_mat) + torch.einsum('bfn,fv->bvn', vn2, self.n2_mat) + torch.einsum('bfn,fv->bvn', vn3, self.n3_mat)) / self.fv_total.unsqueeze(0).unsqueeze(-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    return normals

if __name__ == "__main__":
  import numpy as np
  hand_model = HandModel(root_rot_mode='axisang')
  
  z = torch.normal(mean=0, std=1, size=[10, hand_model.code_length], requires_grad=True).float().cuda() * 0.01
  hand_verts = hand_model.get_vertices(z).detach().cpu().numpy()
  palm_vector = hand_model.palm_direction(z).detach().cpu().numpy()
  back_vector = hand_model.back_direction(z).detach().cpu().numpy()
  side_vector = hand_model.side_direction(z).detach().cpu().numpy()

  import plotly 
  import plotly.graph_objects as go

  for i in range(10):
    x,y,z = hand_verts[i,:,:].mean(0)
    fig = plotly.tools.make_subplots(specs=[[{'type':'surface'}]])
    fig.append_trace(go.Mesh3d(
      x=hand_verts[i,:,0], y=hand_verts[i,:,1], z=hand_verts[i,:,2], 
      i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2],
    ), 1, 1)
    fig.append_trace(go.Scatter3d(
      x=(x,x+palm_vector[i,0]), y=(y,y+palm_vector[i,1]), z=(z,z+palm_vector[i,2])
    ), 1, 1)
    fig.append_trace(go.Scatter3d(
      x=(x,x+back_vector[i,0]), y=(y,y+back_vector[i,1]), z=(z,z+back_vector[i,2])
    ), 1, 1)
    fig.append_trace(go.Scatter3d(
      x=(x,x+side_vector[i,0]), y=(y,y+side_vector[i,1]), z=(z,z+side_vector[i,2])
    ), 1, 1)
    fig.show()
    input()

  # z = torch.normal(mean=0, std=1, size=[1, hand_model.code_length], requires_grad=True).float().cuda()
  # z[:] = 0
  # v = hand_model.get_vertices(z).detach().cpu().numpy()
  # faces = hand_model.layer.th_faces.detach().cpu().numpy() + 1
  # f = open('mano.obj', 'w')
  # for x, y, z in v[0]:
  #   f.write('v %f %f %f\n'%(x,y,z))
  # for i, j, k in faces:
  #   f.write('f %d %d %d\n'%(i,j,k))
  # f.close()

  # rot = robust_compute_rotation_matrix_from_ortho6d(z[:, 3:9])
  # d1 = torch.tensor(np.array([1,0,0]).reshape([1, 3, 1])).float().cuda()
  # d1 = torch.bmm(rot, d1).squeeze().detach().cpu().numpy()
  # d2 = torch.tensor(np.array([0,1,0]).reshape([1, 3, 1])).float().cuda()
  # d2 = torch.bmm(rot, d2).squeeze().detach().cpu().numpy()
  # d3 = torch.tensor(np.array([0,0,1]).reshape([1, 3, 1])).float().cuda()
  # d3 = torch.bmm(rot, d3).squeeze().detach().cpu().numpy()

  # cx, cy, cz = v[0].mean(0)

  # import matplotlib.pyplot as plt
  # from mpl_toolkits.mplot3d import Axes3D
  # ax = plt.subplot(111, projection='3d')
  # ax.scatter(v[0,:,0], v[0,:,1], v[0,:,2], s=2, c='black')
  # ax.quiver(cx, cy, cz, d1[0], d1[1], d1[2], color='red', length=0.3)
  # ax.quiver(cx, cy, cz, d2[0], d2[1], d2[2], color='green', length=0.3)
  # ax.quiver(cx, cy, cz, d3[0], d3[1], d3[2], color='blue', length=0.3)
  # ax.set_xlim([cx-1,cx+1])
  # ax.set_ylim([cy-1,cy+1])
  # ax.set_zlim([cz-1,cz+1])
  # plt.show()
  # print(g)