import time
import pickle

useMaximalCoordinates = 0
import trimesh as tm
from ObjectModel import ObjectModel
from HandModel import HandModel
from CodeUtil import *
import numpy as np
import os
import torch

for meshi in range(4, 7):

  object_model = ObjectModel()
  old_data = pickle.load(open("./logs/"+str(meshi)+"/reduced_dist_result.pkl", 'rb'))
  obj_code = old_data[0]
  old_z = old_data[1]
  contact_point_indices = old_data[2]
  #old_energy = old_data[3]
  hand_model = HandModel(flat_hand_mean=False)

  i_item = 10
  def save_mesh_and_points_not_to_anchor():
    old_hand_verts = hand_model.get_vertices(old_z) #.detach().float().cpu()
    surface_normal = hand_model.get_surface_normals(old_z, old_hand_verts)
    
    for i_item in range(len(obj_code)):
      obj_mesh = get_obj_mesh_by_code(obj_code[i_item])
      print(obj_code.shape)
      np.save("./mesh_and_points"+str(meshi)+"/surface_normal_"+str(i_item)+".npy",surface_normal[i_item].detach().float().cpu().numpy())

      vt = old_hand_verts[i_item].reshape([1,778,3])
      #old_hand_verts_numpy = old_hand_verts.float().cpu()
      objcd = obj_code[i_item].reshape([1,256])
      distance = object_model.distance(objcd, vt)
      distance = distance.detach().float().cpu().numpy()
      distance = distance.reshape([778])
      np.save("./mesh_and_points"+str(meshi)+"/distance_"+str(i_item)+".npy",distance)

      penetration_pts = np.where(distance < 0.05)[0]

      pkl_index = 0
      obj_mesh = get_obj_mesh_by_code(obj_code[i_item])
      obj_mesh.export('./mesh_and_points'+str(meshi)+'/temp'+str(pkl_index)+"_"+str(i_item)+".obj")
      tst = np.asarray([[93,39,123 ],[93,123,119],[93,119,118],[93,118,120],[93,120,121],[93,121,109],\
                                  [93,109,80 ],[93,80,79  ],[93,79,122 ],[93,122,215],[93,215,216],[93,216,280],[93,280,240],[93,240,235]]) - 1
      hand_model.faces = np.concatenate([tst, hand_model.faces])
      hand_mesh = tm.Trimesh(vt.detach().float().cpu().numpy().reshape([778,3]), hand_model.faces, process = False)
      hand_mesh.export('./mesh_and_points'+str(meshi)+'/hand_temp'+str(pkl_index)+"_"+str(i_item)+".obj")

  save_mesh_and_points_not_to_anchor()