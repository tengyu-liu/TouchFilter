# compute forward kinematics from joint angle

import numpy as np
import math
import mat_trans as mt

# Input: qpos (29,) as numpy array
# Output: xpos (25,3) and xquat (25,4) as numpy array

def ForwardKinematic(qpos):
    NUM_PARTS = 25
    xpos = np.zeros((NUM_PARTS, 3))
    xquat = np.zeros((NUM_PARTS, 4))
    xpos[0,:] = qpos[0:3].reshape(1,3)
    xquat[0,:] = qpos[3:7].reshape(1,4)
    Rot_world_forearm = mt.quaternion_matrix(xquat[0,:])
    Tran_world_forearm = mt.translation_matrix(xpos[0,:])

    # wristy, rotate axis = [0,1,0], parent: forearm
    wristyPos = np.array([0,0,0])
    Rot_forearm_wristy = mt.rotation_matrix(qpos[7], [0,1,0])
    Quat_forearm_wristy = mt.quaternion_from_matrix(Rot_forearm_wristy)
    xquat[1,:] = mt.quaternion_multiply(xquat[0,:], Quat_forearm_wristy)
    T_world_wristy = np.matmul(Rot_world_forearm, Rot_forearm_wristy)
    xpos[1,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_wristy, np.append(wristyPos,[1]))))[0:3]
    

    # wristx, rotate axis = [0,0,-1], parent: wristy
    wristxPos = np.array([-3.36826e-5, -0.0476452, 0.00203763])
    Rot_wristy_wristx = mt.rotation_matrix(qpos[8], [0,0,-1])
    Quat_wristy_wristx = mt.quaternion_from_matrix(Rot_wristy_wristx)
    xquat[2,:] = mt.quaternion_multiply(xquat[1,:], Quat_wristy_wristx)
    T_world_wristx = np.matmul(T_world_wristy, np.matmul(mt.translation_matrix(wristxPos), Rot_wristy_wristx))
    xpos[2,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_wristy, np.append(wristxPos,[1]))))[0:3]

    # wristz, rotate axis = [1,0,0], parent: wristx
    wristzPos = np.array([0.0001872, -0.03, -0.002094])
    Rot_wristx_wristz = mt.rotation_matrix(qpos[9], [1,0,0])
    Quat_wristx_wristz = mt.quaternion_from_matrix(Rot_wristx_wristz)
    xquat[3,:] = mt.quaternion_multiply(xquat[2,:], Quat_wristx_wristz)
    T_world_wristz = np.matmul(T_world_wristx, np.matmul(mt.translation_matrix(wristzPos), Rot_wristx_wristz))
    xpos[3,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_wristx, np.append(wristzPos,[1]))))[0:3]

    # palm, no rotation, parent: wristz
    palmPos = np.array([0.025625, 0, 0])
    Rot_wristz_palm = mt.rotation_matrix(0, [1,0,0])
    Quat_wristz_palm = mt.quaternion_from_matrix(Rot_wristz_palm)
    xquat[4,:] = mt.quaternion_multiply(xquat[3,:], Quat_wristz_palm)
    T_world_palm = np.matmul(T_world_wristz, np.matmul(mt.translation_matrix(palmPos), Rot_wristz_palm))
    xpos[4,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_wristz, np.append(palmPos,[1]))))[0:3]

    #=====================================THUMB=============================================
    # thumb0, rotate axis = [0,1,0], parent: palm
    thumb0Pos = np.array([0.00835752, -0.0206978, -0.010093])
    thumb0Quat = np.array([0.990237, 0.0412644, -0.0209178, -0.13149])
    Rot_palm_thumb0 = np.matmul(mt.quaternion_matrix(thumb0Quat), mt.rotation_matrix(qpos[10], [0,1,0]))
    Quat_palm_thumb0 = mt.quaternion_from_matrix(Rot_palm_thumb0)
    xquat[5,:] = mt.quaternion_multiply(xquat[4,:], Quat_palm_thumb0)
    T_world_thumb0 = np.matmul(T_world_palm, np.matmul(mt.translation_matrix(thumb0Pos), Rot_palm_thumb0))
    xpos[5,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_palm, np.append(thumb0Pos,[1]))))[0:3]

    # thumb1, rotate axis = [0,0,-1], parent: thumb0
    thumb1Pos = np.array([0.0209172, -0.00084, 0.0014476])
    thumb1Quat = np.array([1,0,0,0])
    Rot_thumb0_thumb1 = np.matmul(mt.quaternion_matrix(thumb1Quat), mt.rotation_matrix(qpos[11], [0,0,-1]))
    Quat_thumb0_thumb1 = mt.quaternion_from_matrix(Rot_thumb0_thumb1)
    xquat[6,:] = mt.quaternion_multiply(xquat[5,:], Quat_thumb0_thumb1)
    T_world_thumb1 = np.matmul(T_world_thumb0, np.matmul(mt.translation_matrix(thumb1Pos), Rot_thumb0_thumb1))
    xpos[6,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_thumb0, np.append(thumb1Pos,[1]))))[0:3]

    # thumb2, rotate axis = [0,0,-1], parent: thumb1
    thumb2Pos = np.array([0.0335, 0, -0.0007426])
    thumb2Quat = np.array([1,0,0,0])
    Rot_thumb1_thumb2 = np.matmul(mt.quaternion_matrix(thumb2Quat), mt.rotation_matrix(qpos[12], [0,0,-1]))
    Quat_thumb1_thumb2 = mt.quaternion_from_matrix(Rot_thumb1_thumb2)
    xquat[7,:] = mt.quaternion_multiply(xquat[6,:], Quat_thumb1_thumb2)
    T_world_thumb2 = np.matmul(T_world_thumb1, np.matmul(mt.translation_matrix(thumb2Pos), Rot_thumb1_thumb2))
    xpos[7,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_thumb1, np.append(thumb2Pos,[1]))))[0:3]

    # thumb3, rotate axis = [0,0,-1], parent: thumb1
    thumb3Pos = np.array([0.0335, 0, 0.0010854])
    thumb3Quat = np.array([1,0,0,0])
    Rot_thumb2_thumb3 = np.matmul(mt.quaternion_matrix(thumb3Quat), mt.rotation_matrix(qpos[13], [0,0,-1]))
    Quat_thumb2_thumb3 = mt.quaternion_from_matrix(Rot_thumb2_thumb3)
    xquat[8,:] = mt.quaternion_multiply(xquat[7,:], Quat_thumb2_thumb3)
    T_world_thumb3 = np.matmul(T_world_thumb2, np.matmul(mt.translation_matrix(thumb3Pos), Rot_thumb2_thumb3))
    xpos[8,:] = (np.matmul(Tran_world_forearm, np.matmul(T_world_thumb2, np.append(thumb3Pos,[1]))))[0:3]

    #=====================================INDEX=============================================
    # index0, rotate axis = [0,0,1], parent: palm
    xpos[9,:], xquat[9,:], T_palm_index0 = forward_kinematics(PartPos = np.array([0.00986485, -0.0658, 0.00101221]),
                                                              PartQuat = np.array([0.996195, 0, 0.0871557, 0]), 
                                                              q = qpos[14], 
                                                              RotAxis = [0,0,1], 
                                                              T_parent = T_world_palm, 
                                                              xquat_parent = xquat[4,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # index1, rotate axis = [1,0,0], parent: index0
    xpos[10,:], xquat[10,:], T_index0_index1 = forward_kinematics(PartPos = np.array([6.26e-005, -0.018, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[15], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_palm_index0, 
                                                              xquat_parent = xquat[9,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # index2, rotate axis = [1,0,0], parent: index1
    xpos[11,:], xquat[11,:], T_index1_index2 = forward_kinematics(PartPos = np.array([0.001086, -0.0435, 0.0005]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[16], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_index0_index1, 
                                                              xquat_parent = xquat[10,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # index3, rotate axis = [1,0,0], parent: index2
    xpos[12,:], xquat[12,:], T_index2_index3 = forward_kinematics(PartPos = np.array([-0.000635, -0.0245, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[17], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_index1_index2, 
                                                              xquat_parent = xquat[11,:], 
                                                              Tran_world_forearm = Tran_world_forearm)


    #=====================================MIDDLE=============================================
    # middle0, rotate axis = [0,0,1], parent: palm
    xpos[13,:], xquat[13,:], T_palm_middle0 = forward_kinematics(PartPos = np.array([-0.012814, -0.0779014, 0.00544608]),
                                                              PartQuat = np.array([-3.14, 0.0436194, 0, 0]), 
                                                              q = 0, 
                                                              RotAxis = [0,0,-1], 
                                                              T_parent = T_world_palm, 
                                                              xquat_parent = xquat[4,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # middle1, rotate axis = [1,0,0], parent: middle0
    xpos[14,:], xquat[14,:], T_middle0_middle1 = forward_kinematics(PartPos = np.array([6.26e-005, -0.018, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[18], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_palm_middle0, 
                                                              xquat_parent = xquat[13,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # middle2, rotate axis = [1,0,0], parent: middle1
    xpos[15,:], xquat[15,:], T_middle1_middle2 = forward_kinematics(PartPos = np.array([0.001086, -0.0435, 0.0005]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[19], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_middle0_middle1, 
                                                              xquat_parent = xquat[14,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # middle3, rotate axis = [1,0,0], parent: middle2
    xpos[16,:], xquat[16,:], T_middle2_middle3 = forward_kinematics(PartPos = np.array([-0.000635, -0.0245, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[20], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_middle1_middle2, 
                                                              xquat_parent = xquat[15,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    #=====================================RING=============================================
    # ring0, rotate axis = [0,0,-1], parent: palm
    xpos[17,:], xquat[17,:], T_palm_ring0 = forward_kinematics(PartPos = np.array([-0.0354928, -0.0666999, 0.00151221]),
                                                              PartQuat = np.array([0.996195, 0, -0.0871557, 0]), 
                                                              q = qpos[21], 
                                                              RotAxis = [0,0,-1], 
                                                              T_parent = T_world_palm, 
                                                              xquat_parent = xquat[4,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # ring1, rotate axis = [1,0,0], parent: ring0
    xpos[18,:], xquat[18,:], T_ring0_ring1 = forward_kinematics(PartPos = np.array([6.26e-005, -0.018, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[22], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_palm_ring0, 
                                                              xquat_parent = xquat[17,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # ring2, rotate axis = [1,0,0], parent: ring1
    xpos[19,:], xquat[19,:], T_ring1_ring2 = forward_kinematics(PartPos = np.array([0.001086, -0.0435, 0.0005]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[23], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_ring0_ring1, 
                                                              xquat_parent = xquat[18,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # ring3, rotate axis = [1,0,0], parent: ring2
    xpos[20,:], xquat[20,:], T_ring2_ring3 = forward_kinematics(PartPos = np.array([-0.000635, -0.0245, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[24], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_ring1_ring2, 
                                                              xquat_parent = xquat[19,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    #=====================================LITTLE=============================================
    # little0, rotate axis = [0,0,-1], parent: palm
    xpos[21,:], xquat[21,:], T_palm_little0 = forward_kinematics(PartPos = np.array([-0.0562459, -0.0554001, -0.00563858]),
                                                              PartQuat = np.array([0.996195, 0, -0.0871557, 0]), 
                                                              q = qpos[25], 
                                                              RotAxis = [0,0,-1], 
                                                              T_parent = T_world_palm, 
                                                              xquat_parent = xquat[4,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # little1, rotate axis = [1,0,0], parent: little0
    xpos[22,:], xquat[22,:], T_little0_little1 = forward_kinematics(PartPos = np.array([6.26e-005, -0.0178999, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[26], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_palm_little0, 
                                                              xquat_parent = xquat[21,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # little2, rotate axis = [1,0,0], parent: little1
    xpos[23,:], xquat[23,:], T_little1_little2 = forward_kinematics(PartPos = np.array([0.000578, -0.033, 0.0005]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[27], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_little0_little1, 
                                                              xquat_parent = xquat[22,:], 
                                                              Tran_world_forearm = Tran_world_forearm)

    # little3, rotate axis = [1,0,0], parent: little2
    xpos[24,:], xquat[24,:], T_little2_little3 = forward_kinematics(PartPos = np.array([-4.78e-005, -0.0175, 0]),
                                                              PartQuat = np.array([1,0,0,0]), 
                                                              q = qpos[28], 
                                                              RotAxis = [1,0,0], 
                                                              T_parent = T_little1_little2, 
                                                              xquat_parent = xquat[23,:], 
                                                              Tran_world_forearm = Tran_world_forearm)


    return xpos, xquat

def forward_kinematics(PartPos, PartQuat, q, RotAxis, T_parent, xquat_parent, Tran_world_forearm):
    Quat = mt.quaternion_multiply(PartQuat, mt.quaternion_from_matrix(mt.rotation_matrix(q, RotAxis)))#np.matmul(mt.quaternion_matrix(PartQuat), mt.rotation_matrix(q, RotAxis))
    Quat = Quat/np.linalg.norm(Quat)
    Rot = mt.quaternion_matrix(Quat)
    #Quat = mt.quaternion_from_matrix(Rot)
    xquat = mt.quaternion_multiply(xquat_parent, Quat)
    T_child= np.matmul(T_parent, np.matmul(mt.translation_matrix(PartPos), Rot))
    xpos = (np.matmul(Tran_world_forearm, np.matmul(T_parent, np.append(PartPos,[1]))))[0:3]

    return xpos, xquat, T_child

# def rot_from_DH(d, theta, r, alpha):
#     return np.array([[math.cos(theta), -math.sin(theta)*math.cos(alpha),  math.sin(theta)*math.sin(alpha), r*math.cos(theta)],
#                      [math.sin(theta),  math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), r*math.sin(theta)],
#                      [              0,                  math.sin(alpha),                  math.cos(alpha),                 d],
#                      [              0,                                0,                                0,                 1]])
#     # np.array([[math.cos(phi_1), -math.sin(phi_1), 0, a_0],
#     #                  [math.sin(phi_1)*math.cos(alpha_0), math.cos(phi_1)*math.cos(alpha_0), -math.sin(alpha_0), -d_1*math.sin(alpha_0)],
#     #                  [math.sin(phi_1)*math.sin(alpha_0), math.cos(phi_1)*math.sin(alpha_0), -math.cos(alpha_0), -d_1*math.cos(alpha_0)],
#     #                  [0,0,0,1]])

if __name__ == '__main__':
    test = 2
    if test == 0:
        qpos = np.array([-0.0205945749473616,
        -0.438089973792596, 0.222009756049780, 0.0578977637519208,
        -0.00247456469897782, -0.187188940173763, 0.980613086879173,
        -0.00499547906113413, 0.0874440912859860, 0.00524885353414755,
        0.207697004573901, 0.00295159225158279, 0.00128560355778968,
        0.000326174530936420, 0.0477336195577287, 0.0215352767822037,
        0.0179082167239799, 0.0184447927696214, 0.0246841347448013,
        0.0205413430423080, 0.0211652167432111, 5.60702739305758e-05,
        0.0197753700871910, 0.0164541493569334, 0.0169546673692031,
        0.000105879130169245, 0.0127488100446999, 0.0105947618497404,
        0.0109101139659364])
    elif test == 1:
        qpos = np.array([0,
        -0.35, 0.3, 3.27e-7,
        0, 0, 1,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,
        0,0,0,0])
    elif test == 2:
        qpos = np.array([0.00781721574144868,
                        -0.550020903060105,
                        0.112701834554395,
                        -0.0654260216715018,
                        0.432828744818427,
                        -0.346698470873955,
                        0.829565478814546,
                        0.0102918022428136,
                        0.686061263024194,
                        0.385261524545475,
                        1.16306303257058,
                        0.701558693423591,
                        0.636108824372553,
                        0.768419316308683,
                        0.121307429139867,
                        0.696915297312086,
                        0.697237208100826,
                        0.796329106088090,
                        1.05994084317066,
                        1.05591182678686,
                        1.20477947552419,
                        0.173768088766900,
                        1.12097616356151,
                        1.11073895380711,
                        1.26640822677043,
                        0.346355044982957,
                        1.05937028802413,
                        1.04623440960789,
                        1.19027962687426,
                        0.0162105844634238,
                        -0.445184130059323,
                        0.0976391374684536,
                        0.883946104397671,
                        -0.294900077207755,
                        0.0341677841212432,
                        0.361255853255399])
    xpos, xquat = ForwardKinematic(qpos)
    print(xpos)
    print(xquat)