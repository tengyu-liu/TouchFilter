import os
import pickle
import time
from time import sleep

import numpy as np
import pybullet as p
import torch
import trimesh as tm

from ObjectModel import ObjectModel


success = 0
fail = 0 
object_model = ObjectModel()

for mesh_i in range(4,7):
    
  old_data = pickle.load(open("./logs/"+str(mesh_i)+"/reduced_dist_result.pkl", 'rb'))
  obj_code = old_data[0]
  old_z = old_data[1]
  contact_point_indices = old_data[2]

  length = 77
  if (mesh_i == 3 or mesh_i == 6):
    length = 75
  else:
    length = 77
    
  sn_scale = 0.02
  the_scale = 1
  cup_scale = 1
  print(the_scale)
  p.connect(p.GUI)

  i_item = 0

  while(i_item < length):
    delta_xs = []
    p.resetSimulation()
    pkl_index = 0

    objcd = obj_code[i_item].reshape([1,256])
    
    hand_mesh = tm.load_mesh('./mesh_and_points'+str(mesh_i)+'/hand_temp'+str(pkl_index)+"_"+str(i_item)+".obj", process = False)
    surface_normal = np.load("./mesh_and_points"+str(mesh_i)+"/surface_normal_"+str(i_item)+".npy")
    hand_mesh.vertices += sn_scale*surface_normal
    hand_mesh.export('./mesh_and_points'+str(mesh_i)+'/hand_temp_sn'+str(pkl_index)+"_"+str(i_item)+".obj")
  
    vt = torch.tensor(hand_mesh.vertices).reshape([1,778,3]).float().cuda()
    distance = object_model.distance(objcd, vt)
    distance = distance.detach().float().cpu().numpy().reshape([778])
    penetration_pts = np.where(distance < 0.02)[0]
    print(len(penetration_pts))
    
    p.setGravity(0, 0, -10)

    bunnyId = p.loadSoftBody(
      './mesh_and_points'+str(mesh_i)+'/hand_temp_sn'+str(pkl_index)+"_"+str(i_item)+".obj", scale=the_scale,
      useBendingSprings=True, useFaceContact=True,springBendingStiffness=10,
      basePosition=[0,0,0], mass=500.0, frictionCoeff=100)

    for i in range(778):
      if i not in penetration_pts:
        p.createSoftBodyAnchor(bunnyId ,i,-1,-1)
      else:
        pass
    
    v_cup = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                fileName='./mesh_and_points'+str(mesh_i)+'/temp'+str(pkl_index)+"_"+str(i_item)+".obj",
                                rgbaColor=[0,1,0,1],
                                meshScale = [cup_scale,cup_scale,cup_scale]
    )
    c_cup = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                            fileName='./mesh_and_points'+str(mesh_i)+'/temp'+str(pkl_index)+"_"+str(i_item)+".obj",
                            meshScale = [cup_scale,cup_scale,cup_scale]
    )
    a2 = cup_body = p.createMultiBody(baseMass=7,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=c_cup,
                                    baseVisualShapeIndex=v_cup,
                                    basePosition=[0,0,0],#[0.2, 0.7, 1],
                                    useMaximalCoordinates=True)

    lateral_eps = 5
    epsilon = 100
    p.changeDynamics(a2,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon)
    p.changeDynamics(0,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon)
    p.changeDynamics(1,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon)
    p.changeDynamics(2,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon)

    i = 0
    is_fail = False
    while p.isConnected() and i < 1000:
      i += 1
      p.stepSimulation()
      dx = np.linalg.norm(p.getBasePositionAndOrientation(a2)[0])
      if dx > 0.3:
        is_fail = True
    
    if is_fail:
      fail += 1
    else:
      success += 1

    i_item += 1

print(success, fail)
