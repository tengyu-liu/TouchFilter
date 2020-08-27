import os
import sys
import time
import pybullet as p
import time
import pickle

import numpy as np

success = 0
fail = 0

# p.connect(p.GUI)
p.connect(p.GUI)
p.setGravity(0, 0, -10)

for mesh_i in range(4,7):
  length = 77
  if (mesh_i == 3 or mesh_i == 6):
    length = 75
  else:
    length = 77
    
  cup_scale = 1.2

  i_item = 0
  total_x = []
  the_scales = []
  temp_scale = 1
  temp_x = np.inf
  #for i_item in range(2,77):
  while(i_item < length):
    p.resetSimulation()
    distance = np.load("./mesh_and_points"+str(mesh_i)+"/distance_"+str(i_item)+".npy")
    penetration_pts = np.where(distance < 0.2)[0]
    # print(len(penetration_pts))
    pkl_index = 0

    import os

    import pybullet as p
    from time import sleep
    import pybullet_data

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    bunnyId = p.loadSoftBody(
      './mesh_and_points'+str(mesh_i)+'/hand_temp'+str(pkl_index)+"_"+str(i_item)+".obj", 
      useBendingSprings=True, useFaceContact=True, springBendingStiffness=10, 
      basePosition=[0,0,0], mass=500.0, frictionCoeff=100)#, repulsionStiffness=0)

    for i in range(778):
        if i not in penetration_pts:
            p.createSoftBodyAnchor(bunnyId ,i,-1,-1)
        else:
            continue

    # v_cup = p.createVisualShape(shapeType=p.GEOM_MESH, 
    #                             fileName='./mesh_and_points'+str(mesh_i)+'/hand_temp'+str(pkl_index)+"_"+str(i_item)+".obj",
    #                             rgbaColor=[0,1,0,1],
    #                             # meshScale = [cup_scale,cup_scale,cup_scale]
    # )

    # c_cup = p.createCollisionShape(shapeType=p.GEOM_MESH, 
    #                         fileName='./mesh_and_points'+str(mesh_i)+'/hand_temp'+str(pkl_index)+"_"+str(i_item)+".obj",
    #                         # meshScale = [cup_scale,cup_scale,cup_scale]
    # )

    # bunnyId = p.createMultiBody(baseMass=0,
    #                                 baseInertialFramePosition=[0, 0, 0],
    #                                 baseCollisionShapeIndex=c_cup,
    #                                 baseVisualShapeIndex=v_cup,
    #                                 basePosition=[0,0,0],#[0.2, 0.7, 1],
    #                                 useMaximalCoordinates=True)

    
    v_cup = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                fileName='./mesh_and_points'+str(mesh_i)+'/temp'+str(pkl_index)+"_"+str(i_item)+".obj",
                                rgbaColor=[1,0,0,1],
                                # meshScale = [cup_scale,cup_scale,cup_scale]
    )

    c_cup = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                            fileName='./mesh_and_points'+str(mesh_i)+'/temp'+str(pkl_index)+"_"+str(i_item)+".obj",
                            # meshScale = [cup_scale,cup_scale,cup_scale]
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
      # print('\r%d: %d'%(i_item, i), end='')
      i += 1
      p.stepSimulation()
      time.sleep(0.001)
      dx = np.linalg.norm(p.getBasePositionAndOrientation(a2)[0])
      if dx > 0.3:
        is_fail = True
        break
    # print()
    if is_fail:
      fail += 1
    else:
      success += 1

    i_item += 1
    print(success, fail)
  # np.save("mesh"+str(mesh_i)+"_total_x_scale_"+str(the_scale)+"1e2mass7.npy",total_x)

p.disconnect()
print(success, fail)