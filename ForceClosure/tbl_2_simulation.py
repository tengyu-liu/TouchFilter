import sys
import time
import os
import trimesh as tm
import pickle
import numpy as np
import torch
from HandModel import HandModel
from ObjectModel import ObjectModel
import pybullet as pb
from CodeUtil import *

hand_model = HandModel(
  root_rot_mode='ortho6d', 
  robust_rot=False,
  flat_hand_mean=False,
  n_contact=5)
object_model = ObjectModel()

obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = pickle.load(open('data/final_optim.pkl', 'rb'))

linear_independence = torch.tensor(linear_independence).float()
force_closure = torch.tensor(force_closure).float()
surface_distance = torch.tensor(surface_distance).float()
penetration = torch.tensor(penetration).float()
z_norm = torch.tensor(z_norm).float()
normal_alignment = torch.tensor(normal_alignment).float()

pb.connect(pb.DIRECT)
# uncomment the following line to visualize each simulation (significantly slower)
# pb.connect(pb.GUI) 
decompose_obj = True

hand_verts = hand_model.get_vertices(z).detach().cpu().numpy()

# make hand mesh water tight
hand_faces = hand_model.faces
hand_ceal = np.asarray([[93,39,123 ],[93,123,119],[93,119,118],[93,118,120],[93,120,121],[93,121,109],\
                            [93,109,80 ],[93,80,79  ],[93,79,122 ],[93,122,215],[93,215,216],[93,216,280],[93,280,240],[93,240,235]]) - 1
hand_faces = np.concatenate([hand_ceal, hand_faces])

results = []
energies = []
 
for i in range(len(z)):
    pb.resetSimulation()
    pb.setGravity(0, 0, -10)
    # load hand
    hand_mesh = tm.Trimesh(hand_verts[i], hand_faces, process=False)
    hand_mesh.export('tmp/hand_%d.obj'%i)
    pb.vhacd('tmp/hand_%d.obj'%i, 'tmp/hand_%d.obj'%i, '/dev/null')
    vh = pb.createVisualShape(shapeType=pb.GEOM_MESH, 
                                fileName='tmp/hand_%d.obj'%i,
                                rgbaColor=[0,1,0,1],
    )
    ch = pb.createCollisionShape(shapeType=pb.GEOM_MESH, 
                            fileName='tmp/hand_%d.obj'%i,
    )
    pb_hand = pb.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=ch,
                                    baseVisualShapeIndex=vh,
                                    basePosition=[0,0,0],
                                    useMaximalCoordinates=False)
    obj_fn = get_obj_fn_by_code(obj_code[i])
    pb.vhacd(obj_fn, obj_fn[:-4] + '_cd.obj', '/dev/null')
    v = pb.createVisualShape(shapeType=pb.GEOM_MESH, 
                                fileName=obj_fn[:-4] + '_cd.obj',
                                rgbaColor=[1,0,0,1],
    )
    c = pb.createCollisionShape(shapeType=pb.GEOM_MESH, 
                            fileName=obj_fn[:-4] + '_cd.obj',
    )
    pb_obj = pb.createMultiBody(baseMass=1,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=c,
                                    baseVisualShapeIndex=v,
                                    basePosition=[0,0,0],
                                    useMaximalCoordinates=False)
    # update dynamics
    lateral_eps = 50
    epsilon = 100
    pb.changeDynamics(pb_hand, pb_obj, lateralFriction=lateral_eps, rollingFriction=epsilon, spinningFriction=epsilon)
    pb.changeDynamics(pb_obj, pb_hand, lateralFriction=lateral_eps, rollingFriction=epsilon, spinningFriction=epsilon)
    # simulate
    is_fail = False
    for step in range(500):
        pb.stepSimulation()
        dz = pb.getBasePositionAndOrientation(pb_obj)[0][2]
        # time.sleep(0.001)
        if dz < -0.5:
            is_fail = True
            break
    if is_fail:
        results.append(0)
    else:
        results.append(1)
    print('\r', i, np.mean(results), end='', flush=True, file=sys.stderr)

linear_independence = torch.stack(linear_independence)
force_closure = torch.stack(force_closure)
surface_distance = torch.stack(surface_distance)
penetration = torch.stack(penetration)
z_norm = torch.stack(z_norm)
normal_alignment = torch.stack(normal_alignment)

energy = (surface_distance).detach().cpu().numpy()

idx = np.argsort(energy)

success_rates = []
for i in range(1, len(idx)+1):
  success_rates.append(results[idx[:i]].mean())

for i in range(100):
  print(i, idx[i], success_rates[i], energy[idx[i]])

def find(th):
  for i in range(len(idx)):
    if energy[idx[i]] > th:
      return success_rates[i-1]
  return success_rates[-1]
    
for th in [0.0005,0.0015,0.0025,0.0035,0.0045]:
  print(th, find(th))
