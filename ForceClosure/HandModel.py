from collections import defaultdict
from manopth.manolayer import ManoLayer
from manopth.rot6d import robust_compute_rotation_matrix_from_ortho6d
from manopth.rodrigues_layer import batch_rodrigues
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

    # FIXME: find the three vectors by locating vertices on palm
    self.forward_base_ids = [34,268]
    self.sideway_base_ids = [96,115]
    self.facedir_base_ids = [218,95]

    self.texture_coords = torch.tensor(self.get_texture_coords().reshape([1, 1, -1, 2]) * 2 - 1).float().cuda()
    self.faces = self.layer.th_faces.detach().cpu().numpy()
    self.num_points = self.texture_coords.shape[2]
    self.verts_eye = torch.tensor(np.eye(self.num_points)).float().cuda()
    self.n1_mat = self.verts_eye[self.faces[:,0]]   # F x V
    self.n2_mat = self.verts_eye[self.faces[:,1]]   # F x V
    self.n3_mat = self.verts_eye[self.faces[:,2]]   # F x V
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
  from ObjectModel import ObjectModel
  from CodeUtil import *
  from PenetrationModel import PenetrationModel

  obj_code, obj_idx = get_obj_code_random(1)
  obj_mesh = get_obj_mesh(obj_idx[0])

  hand_model = HandModel(root_rot_mode='rot6d')

  object_model = ObjectModel()
  penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

  while True:
    z = torch.normal(mean=0, std=1, size=[1, hand_model.code_length], requires_grad=True).float().cuda() * 0.5
    hand_verts = hand_model.get_vertices(z)
    hand_verts_bak = hand_verts.clone()
    back_direction = hand_model.back_direction(z)
    palm_point = hand_verts[:, [hand_model.facedir_base_ids[1]], :]
    palm_normal = object_model.gradient(palm_point, object_model.distance(obj_code, palm_point))[:,0,:]
    palm_normal = palm_normal / torch.norm(palm_normal, dim=-1, keepdim=True)
    print(torch.dot(palm_normal[0], back_direction[0]))
    z2 = hand_model.align(z, back_direction, palm_normal)
    hand_verts = hand_model.get_vertices(z2)
    back_direction = hand_model.back_direction(z2)
    palm_point = hand_verts[:, [hand_model.facedir_base_ids[1]], :]
    palm_normal = object_model.gradient(palm_point, object_model.distance(obj_code, palm_point))[:,0,:]
    palm_normal = palm_normal / torch.norm(palm_normal, dim=-1, keepdim=True)
    print(torch.dot(palm_normal[0], back_direction[0]))
    z2 = hand_model.align(z2, back_direction, palm_normal)
    hand_verts = hand_model.get_vertices(z2)
    back_direction = hand_model.back_direction(z2)
    palm_point = hand_verts[:, [hand_model.facedir_base_ids[1]], :]
    palm_normal = object_model.gradient(palm_point, object_model.distance(obj_code, palm_point))[:,0,:]
    palm_normal = palm_normal / torch.norm(palm_normal, dim=-1, keepdim=True)
    print(torch.dot(palm_normal[0], back_direction[0]))
    z2 = hand_model.align(z2, back_direction, palm_normal)
    hand_verts = hand_model.get_vertices(z2)
    back_direction = hand_model.back_direction(z2)
    palm_point = hand_verts[:, [hand_model.facedir_base_ids[1]], :]
    palm_normal = object_model.gradient(palm_point, object_model.distance(obj_code, palm_point))[:,0,:]
    palm_normal = palm_normal / torch.norm(palm_normal, dim=-1, keepdim=True)
    print(torch.dot(palm_normal[0], back_direction[0]))



    max_penetration = penetration_model.get_max_penetration(obj_code, z2)
    print(max_penetration)
    z3 = z2.clone()
    z2[:,:3] += back_direction * max_penetration

    print(penetration_model.get_max_penetration(obj_code, z2))

    hand_verts = hand_verts_bak.detach().cpu().numpy()
    hand_verts2 = hand_model.get_vertices(z2).detach().cpu().numpy()
    hand_verts3 = hand_model.get_vertices(z3).detach().cpu().numpy()

    import plotly 
    import plotly.graph_objects as go

    fig = plotly.tools.make_subplots(specs=[[{'type':'surface'}]])

    i_item = 0

    # px = torch.tensor([0,1, 0,0, 0,0, 0]).float().cuda()
    # py = torch.tensor([0,0, 0,1, 0,0, 0]).float().cuda()
    # pz = torch.tensor([0,0, 0,0, 0,1, 0]).float().cuda()

    # xyz = torch.stack([px,py,pz], dim=-1)
    # rot1 = robust_compute_rotation_matrix_from_ortho6d(z[:,3:9])[0]
    # rot2 = robust_compute_rotation_matrix_from_ortho6d(z2[:,3:9])[0]
    # r1p = torch.matmul(rot1, xyz.transpose(0,1)).transpose(0,1).detach().cpu().numpy()
    # r2p = torch.matmul(rot2, xyz.transpose(0,1)).transpose(0,1).detach().cpu().numpy()
    # axis = forward_direction[0]

    # fig.append_trace(go.Scatter3d(
    #   x=r1p[:,0], y=r1p[:,1], z=r1p[:,2], marker=dict(color='red')
    # ), 1, 1)

    # fig.append_trace(go.Scatter3d(
    #   x=r2p[:,0], y=r2p[:,1], z=r2p[:,2], marker=dict(color='blue')
    # ), 1, 1)

    # fig.append_trace(go.Scatter3d(
    #   x=[-axis[0],axis[0]], y=[-axis[1], axis[1]], z=[-axis[2], axis[2]]
    # ), 1, 1)

    fig.append_trace(go.Mesh3d(
      x=hand_verts[i_item,:,0], y=hand_verts[i_item,:,1], z=hand_verts[i_item,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      color='lightpink', opacity=1
    ), 1, 1)

    fig.append_trace(go.Mesh3d(
      x=hand_verts2[i_item,:,0], y=hand_verts2[i_item,:,1], z=hand_verts2[i_item,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      color='red', opacity=1
    ), 1, 1)

    fig.append_trace(go.Mesh3d(
      x=hand_verts3[i_item,:,0], y=hand_verts3[i_item,:,1], z=hand_verts3[i_item,:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2], 
      color='brown', opacity=1
    ), 1, 1)

    fig.append_trace(go.Mesh3d(
      x=obj_mesh.vertices[:,0], y=obj_mesh.vertices[:,1], z=obj_mesh.vertices[:,2], 
      i=obj_mesh.faces[:,0], j=obj_mesh.faces[:,1], k=obj_mesh.faces[:,2], 
      color='blue', opacity=1
    ), 1, 1)
    fig.update_layout(dict(scene=dict(aspectmode='data')), showlegend=False)

    fig.show()
    input()

  # cx, cy, cz = hand_verts[0].mean(0)

  # fig.append_trace(go.Cone(
  #   x=(cx,), y=(cy,), z=(cz,), 
  #   u=(back_direction[0,0],), 
  #   v=(back_direction[0,1],), 
  #   w=(back_direction[0,2],), 
  #   ), 1, 1)

  # fig.show()

  # for i in range(hand_verts.shape[1]):
  #   fig.append_trace(go.Cone(
  #     x=(hand_verts[0,i,0],), y=(hand_verts[0,i,1],), z=(hand_verts[0,i,2],),
  #     u=(hand_normal[0,i,0],), v=(hand_normal[0,i,1],), w=(hand_normal[0,i,2],),
  #     showscale=False, sizemode='absolute', sizeref=0.05
  #   ), 1, 1)
  # fig.show()

  # for i in range(10):
  #   x,y,z = hand_verts[i,:,:].mean(0)
  #   fig = plotly.tools.make_subplots(specs=[[{'type':'surface'}]])
  #   fig.append_trace(go.Mesh3d(
  #     x=hand_verts[i,:,0], y=hand_verts[i,:,1], z=hand_verts[i,:,2], 
  #     i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2],
  #   ), 1, 1)
  #   fig.append_trace(go.Scatter3d(
  #     x=(x,x+palm_vector[i,0]), y=(y,y+palm_vector[i,1]), z=(z,z+palm_vector[i,2])
  #   ), 1, 1)
  #   fig.append_trace(go.Scatter3d(
  #     x=(x,x+back_vector[i,0]), y=(y,y+back_vector[i,1]), z=(z,z+back_vector[i,2])
  #   ), 1, 1)
  #   fig.append_trace(go.Scatter3d(
  #     x=(x,x+side_vector[i,0]), y=(y,y+side_vector[i,1]), z=(z,z+side_vector[i,2])
  #   ), 1, 1)
  #   fig.show()
  #   input()

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