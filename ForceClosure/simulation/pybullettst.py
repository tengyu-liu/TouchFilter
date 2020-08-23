
import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)

p.setGravity(0, 0, -10)


v_cup = p.createVisualShape(shapeType=p.GEOM_MESH, 
                            fileName="temp2500_10.obj",
                            rgbaColor=[0,1,0,1]
)

c_cup = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                        fileName="temp2500_10.obj"
)

a2 = cup_body = p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=c_cup,
                                baseVisualShapeIndex=v_cup,
                                basePosition=[0,0,1],#[0.2, 0.7, 1],
                                useMaximalCoordinates=True)

v_shape_index = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                rgbaColor=[1,0,0,1],
                                fileName="hand_temp2500_10.obj"
        )
c_shape_index = p.createCollisionShape(shapeType=p.GEOM_MESH, 
                        fileName="hand_temp2500_10.obj"
)
a1 = p.createMultiBody(baseMass=0,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=c_shape_index,
                                baseVisualShapeIndex=v_shape_index,
                                basePosition=[0,0,1],
                                useMaximalCoordinates=True)

lateral_eps = 3
epsilon = 10
p.changeDynamics(a1,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon, )
p.changeDynamics(a2,-1,lateralFriction = lateral_eps, rollingFriction=epsilon, spinningFriction= epsilon, )

i = 0
while p.isConnected() or i < 1000:
  i += 1
  p.stepSimulation()
  contact_points = p.getContactPoints(cup_body)
  print(contact_points)
  p.setCollisionFilterPair(a1, a2, -1, -1, 0)
  p.setCollisionFilterPair(a1, a2, -1, -1, 1)
  input()
