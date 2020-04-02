import os

import numpy as np
import pybullet as pb

import mat_trans as mt
from forward_kinematics import ForwardKinematic

class Simulator:
    def __init__(self):
        self.shape_dir = os.path.join(os.path.dirname(__file__), '../../data/hand')
        self.parts = ['palm', 
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']
        pass
    
    @staticmethod
    def rotation_matrix(rot):
        a1 = rot[:,0]
        a2 = rot[:,1]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        b3 = np.cross(b1, b2)
        eye = np.eye(4)
        eye[:3,:3] = np.stack([b1, b2, b3], axis=-1)
        return eye

    def connect(self, with_gui=True):
        if with_gui:
            self.physicsClient = pb.connect(pb.GUI)
        else:
            self.physicsClient = pb.connect(pb.DIRECT)

    def simulate(self):
        pb.stepSimulation()
        pass
    
    def get_cup_position_orientation(self):
        return pb.getBasePositionAndOrientation(self.cup_body)
    
    def set_gravity(self, g):
        pb.setGravity(g[0], g[1], g[2])

    def generate_world(self, hand_z):
        pb.setGravity(0,0,-10)
        jrot = hand_z[:22]
        grot = np.reshape(hand_z[22:28], [3, 2])
        gpos = hand_z[28:]
        
        grot = mt.quaternion_from_matrix(Simulator.rotation_matrix(grot))

        qpos = np.concatenate([gpos, grot, jrot])

        xpos, xquat = ForwardKinematic(qpos)
        for i in range(21):
            part_name = self.parts[i]
            quat = xquat[i + 4]
            pos = xpos[i + 4]
            v_shape_index = pb.createVisualShape(shapeType=pb.GEOM_MESH, 
                                    fileName=os.path.join(os.path.dirname(__file__), '../../data/hand/%s.STL'%part_name),
                                    rgbaColor=[1,0,0,1],
                                    visualFramePosition=pos,
                                    visualFrameOrientation=[quat[1], quat[2], quat[3], quat[0]]
            )
            c_shape_index = pb.createCollisionShape(shapeType=pb.GEOM_MESH, 
                                    fileName=os.path.join(os.path.dirname(__file__), '../../data/hand/%s.STL'%part_name),
                                    collisionFramePosition=pos,
                                    collisionFrameOrientation=[quat[1], quat[2], quat[3], quat[0]]
            )
            pb.createMultiBody(baseMass=0,
                                    baseInertialFramePosition=[0, 0, 0],
                                    baseCollisionShapeIndex=c_shape_index,
                                    baseVisualShapeIndex=v_shape_index,
                                    basePosition=[0, 0, 0],
                                    useMaximalCoordinates=True)

        v_cup = pb.createVisualShape(shapeType=pb.GEOM_MESH, 
                                fileName=os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/3.obj'),
                                rgbaColor=[0,1,0,1],
                                # visualFramePosition=[xpos[4,0]+0.1, xpos[4,1], xpos[4,2] + 0.2],
                                # visualFrameOrientation=[quat[1], quat[2], quat[3], quat[0]]
        )
        c_cup = pb.createCollisionShape(shapeType=pb.GEOM_MESH, 
                                fileName=os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/3.obj'),
                                # collisionFramePosition=[xpos[4,0]+0.1, xpos[4,1], xpos[4,2] + 0.2],
                                # collisionFrameOrientation=[quat[1], quat[2], quat[3], quat[0]]
        )
        self.cup_body = pb.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=c_cup,
                                baseVisualShapeIndex=v_cup,
                                basePosition=[0, 0, 0],
                                useMaximalCoordinates=True)

    def reset(self):
        pb.resetSimulation()

    def disconnect(self):
        pb.disconnect()
    
if __name__ == "__main__":
    import numpy as np
    import time
    s = Simulator()
    s.connect()
    z = np.zeros([29])
    z[3] = 1.0
    z[7:] = np.random.random([22]) * 1
    s.generate_world(z)
    for i in range(100):
        s.simulate()
        time.sleep(1/40.)
    print(s.get_cup_position_orientation())
    s.disconnect()