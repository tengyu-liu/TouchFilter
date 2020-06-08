import numpy as np
import tensorflow as tf

def quaternion_matrix(q):
    q = np.array(q)
    q_shape = q.shape
    if len(q_shape) == 1:
        q = np.expand_dims(q, axis=0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    a = q[:,0]
    b = q[:,1]
    c = q[:,2]
    d = q[:,3]
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b * 2
    ac = a * c * 2
    ad = a * d * 2
    bc = b * c * 2
    bd = b * d * 2
    cd = c * d * 2
    q_mat = np.transpose(np.stack([
                np.stack([a2+b2-c2-d2, bc-ad, bd+ac, ac * 0]),
                np.stack([bc+ad, a2-b2+c2-d2, cd-ab, ac * 0]),
                np.stack([bd-ac, cd+ab, a2-b2-c2+d2, ac * 0]),
                np.stack([ac * 0, ac * 0, ac * 0, ac * 0 + 1])]), [2,0,1])
    
    if len(q_shape) == 1:
        return q_mat[0]
    return q_mat

"""
Converts qpos from translation-quaternion-angle to
    translation - 6dRotationMatrix - 2dRotationMatrix
"""
def preprocess_qpos(qpos: np.ndarray) -> np.ndarray:
    translation = qpos[:,:3]
    rotation = qpos[:,3:7]
    rotation = quaternion_matrix(rotation)[:,:3,:2]
    angles_s = np.sin(qpos[:,7:])
    angles_c = np.cos(qpos[:,7:])
    angles = np.stack([angles_s, angles_c], axis=-1)
    return translation, rotation, angles

def tf_outer(a, b):
    return tf.expand_dims(a, axis=-1) * tf.expand_dims(b, axis=-2)

# Expands a nx3x3 to nx4x4 matrix where the right-bottom element is one
def tf_expand_eye(a):
    return tf.concat([
        tf.concat([
            a, tf.tile(tf.reshape([0.,0.,0.], [1, 1, 3]), [a.shape[0], 1, 1])], axis=-2), 
            tf.tile(tf.reshape([0.,0.,0.,1.], [1, 4, 1]), [a.shape[0], 1, 1])], axis=-1)

def angle_axis_rotation_matrix(angle: tf.Tensor, direction: tf.Tensor) -> tf.Tensor:
    direction = tf.tile(tf.expand_dims(direction, axis=0), [angle.shape[0], 1])

    direction = direction / tf.norm(direction, axis=-1, keepdims=True)
    sina = tf.math.sin(angle)
    cosa = tf.math.cos(angle)

    R = tf.reduce_sum(tf.diag(tf.stack([cosa, cosa, cosa], axis=-1)), axis=2)

    R += tf_outer(direction, direction) * tf.expand_dims(tf.expand_dims((1.0 - cosa), axis=-1), axis=-1)

    direction *= tf.expand_dims(sina, axis=-1)

    zeros = tf.zeros([direction.shape[0]])

    R2 = tf.transpose(tf.stack([
        tf.stack([zeros, zeros - direction[:,2], zeros + direction[:,1]]),
        tf.stack([zeros + direction[:,2], zeros, zeros - direction[:,0]]),
        tf.stack([zeros - direction[:,1], zeros + direction[:,0], zeros])
    ]), [2, 0, 1])

    R += R2

    M = tf_expand_eye(R)
    return M

"""
Input:  rot: N x 3 x 2
Output: mat: N x 4 x 4
"""
def rotation_matrix(rot: tf.Tensor) -> tf.Tensor:
    a1 = rot[:,:,0]
    a2 = rot[:,:,1]
    b1 = a1 / tf.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - tf.reduce_sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / tf.norm(b2, axis=-1, keepdims=True)
    b3 = tf.cross(b1, b2)
    return tf_expand_eye(tf.stack([b1, b2, b3], axis=-1))

"""
Input:  trans: N x 3
Output: mat:   N x 4 x 4
"""
def translation_matrix(trans: tf.Tensor) -> tf.Tensor:
    mat =tf.eye(3, 3, [trans.shape[0]])
    return tf.concat([
            tf.concat([
                mat, tf.reshape(trans, [mat.shape[0], 3, 1])], axis=-1), 
                tf.tile(tf.reshape([0.,0.,0.,1.], [1, 1, 4]), [mat.shape[0], 1, 1])], axis=-2)

"""
Hand Kinematics for TensorFlow

Input:  pos   (N x 3)       root translation
        rot   (N x 3 x 2)   root rotation
        qpos  (N x 22 x 1)  rotation representation of each joint
Output: xpos, xquat  {p: (N x 3)} and {p: (N x 4 x 4)} for each part p

This module converts original latent vector Z to trans-rot config C.
Notice that in this module we assume that the rotations in Z are sin(\theta)
    instead of \theta. Proper pre-processing should be applied beforehand in numpy.
"""
def kinematics(world_translation: tf.Tensor, world_rotation: tf.Tensor, angles: tf.Tensor):
    xpos = {}
    xquat = {}
    Tran_world_forearm = translation_matrix(world_translation)
    Rot_world_forearm = rotation_matrix(world_rotation)

    xpos['forearm'] = world_translation
    xquat['forearm'] = Rot_world_forearm

    wristy_axis = tf.constant([0., 1., 0.])
    wristy_pos1 = tf.tile(tf.reshape(tf.constant([0., 0, 0, 1]), [1, 4, 1]), [angles.shape[0], 1, 1])
    Rot_forearm_wristy = angle_axis_rotation_matrix(angles[:,0], wristy_axis)
    xquat['wristy'] = tf.matmul(xquat['forearm'], Rot_forearm_wristy)
    T_world_wristy = xquat['wristy']
    xpos['wristy'] = tf.matmul(Tran_world_forearm, tf.matmul(T_world_wristy, wristy_pos1))[:,:3, 0]

    wristx_axis = tf.constant([0., 0., -1.])
    wristx_pos = tf.tile(tf.reshape(tf.constant([-3.36826e-5, -0.0476452, 0.00203763]), [1, 3]), [angles.shape[0], 1])
    wristx_pos1 = tf.tile(tf.reshape(tf.constant([-3.36826e-5, -0.0476452, 0.00203763, 1]), [1, 4, 1]), [angles.shape[0], 1, 1])
    Rot_wristy_wristx = angle_axis_rotation_matrix(angles[:,1], wristx_axis)
    xquat['wristx'] = tf.matmul(xquat['wristy'], Rot_wristy_wristx)
    T_world_wristx = tf.matmul(T_world_wristy, tf.matmul(translation_matrix(wristx_pos), Rot_wristy_wristx))
    xpos['wristx'] = tf.matmul(Tran_world_forearm, tf.matmul(T_world_wristy, wristx_pos1))[:,:3,0]

    wristz_axis = tf.constant([1., 0., 0.])
    wristz_pos = tf.tile(tf.reshape(tf.constant([0.0001872, -0.03, -0.002094]), [1, 3]), [angles.shape[0], 1])
    wristz_pos1 = tf.tile(tf.reshape(tf.constant([0.0001872, -0.03, -0.002094, 1]), [1, 4, 1]), [angles.shape[0], 1, 1])
    Rot_wristx_wristz = angle_axis_rotation_matrix(angles[:,2], wristz_axis)
    xquat['wristz'] = tf.matmul(xquat['wristx'], Rot_wristx_wristz)
    T_world_wristz = tf.matmul(T_world_wristx, tf.matmul(translation_matrix(wristz_pos), Rot_wristx_wristz))
    xpos['wristz'] = tf.matmul(Tran_world_forearm, tf.matmul(T_world_wristx, wristz_pos1))[:,:3,0]

    palm_pos = tf.tile(tf.reshape(tf.constant([0.025625, 0., 0.]), [1, 3]), [angles.shape[0], 1])
    palm_pos1 = tf.tile(tf.reshape(tf.constant([0.025625, 0., 0., 1]), [1, 4, 1]), [angles.shape[0], 1, 1])
    xquat['palm'] = xquat['wristz']
    T_world_palm = tf.matmul(T_world_wristz, translation_matrix(palm_pos))
    xpos['palm'] = tf.matmul(Tran_world_forearm, tf.matmul(T_world_wristz, palm_pos1))[:,:3,0]

    #=====================================THUMB=============================================
    xpos['thumb0'], xquat['thumb0'], T_palm_thumb0 = forward_kinematics(
        [0.00835752, -0.0206978, -0.010093], 
        [0.990237, 0.0412644, -0.0209178, -0.13149], 
        angles[:,3], [0., 1., 0.], T_world_palm, xquat['palm'], Tran_world_forearm)

    xpos['thumb1'], xquat['thumb1'], T_thumb0_thumb1 = forward_kinematics(
        [0.0209172, -0.00084, 0.0014476], 
        None,
        angles[:,4], [0., 0., -1.], T_palm_thumb0, xquat['thumb0'], Tran_world_forearm)

    xpos['thumb2'], xquat['thumb2'], T_thumb1_thumb2 = forward_kinematics(
        [0.0335, 0, -0.0007426], 
        None,
        angles[:,5], [0., 0., -1.], T_thumb0_thumb1, xquat['thumb1'], Tran_world_forearm)

    xpos['thumb3'], xquat['thumb3'], T_thumb2_thumb3 = forward_kinematics(
        [0.0335, 0, 0.0010854], 
        None,
        angles[:,6], [0., 0., -1.], T_thumb1_thumb2, xquat['thumb2'], Tran_world_forearm)

    #=====================================INDEX=============================================
    xpos['index0'], xquat['index0'], T_palm_index0 = forward_kinematics(
        [0.00986485, -0.0658, 0.00101221], 
        [0.996195, 0, 0.0871557, 0], 
        angles[:,7], [0., 0, 1], T_world_palm, xquat['palm'], Tran_world_forearm)

    xpos['index1'], xquat['index1'], T_index0_index1 = forward_kinematics(
        [6.26e-005, -0.018, 0], 
        None,
        angles[:,8], [1., 0, 0], T_palm_index0, xquat['index0'], Tran_world_forearm)

    xpos['index2'], xquat['index2'], T_index1_index2 = forward_kinematics(
        [0.001086, -0.0435, 0.0005], 
        None,
        angles[:,9], [1., 0, 0], T_index0_index1, xquat['index1'], Tran_world_forearm)

    xpos['index3'], xquat['index3'], T_index2_index3 = forward_kinematics(
        [-0.000635, -0.0245, 0], 
        None,
        angles[:,10], [1., 0, 0], T_index1_index2, xquat['index2'], Tran_world_forearm)

    #=====================================MIDDLE=============================================
    zero_angle = tf.zeros(angles.shape[0])
    xpos['middle0'], xquat['middle0'], T_palm_middle0 = forward_kinematics(
        [-0.012814, -0.0779014, 0.00544608], 
        [-3.14, 0.0436194, 0, 0], 
        zero_angle, [0., 0, -1], T_world_palm, xquat['palm'], Tran_world_forearm)

    xpos['middle1'], xquat['middle1'], T_middle0_middle1 = forward_kinematics(
        [6.26e-005, -0.018, 0], 
        None,
        angles[:,11], [1., 0, 0], T_palm_middle0, xquat['middle0'], Tran_world_forearm)

    xpos['middle2'], xquat['middle2'], T_middle1_middle2 = forward_kinematics(
        [0.001086, -0.0435, 0.0005], 
        None,
        angles[:,12], [1., 0, 0], T_middle0_middle1, xquat['middle1'], Tran_world_forearm)

    xpos['middle3'], xquat['middle3'], T_middle2_middle3 = forward_kinematics(
        [-0.000635, -0.0245, 0], 
        None,
        angles[:,13], [1., 0, 0], T_middle1_middle2, xquat['middle2'], Tran_world_forearm)

    #=====================================RING=============================================
    xpos['ring0'], xquat['ring0'], T_palm_ring0 = forward_kinematics(
        [-0.0354928, -0.0666999, 0.00151221], 
        [0.996195, 0, -0.0871557, 0], 
        angles[:,14], [0., 0, -1], T_world_palm, xquat['palm'], Tran_world_forearm)

    xpos['ring1'], xquat['ring1'], T_ring0_ring1 = forward_kinematics(
        [6.26e-005, -0.018, 0], 
        None,
        angles[:,15], [1., 0, 0], T_palm_ring0, xquat['ring0'], Tran_world_forearm)

    xpos['ring2'], xquat['ring2'], T_ring1_ring2 = forward_kinematics(
        [0.001086, -0.0435, 0.0005], 
        None,
        angles[:,16], [1., 0, 0], T_ring0_ring1, xquat['ring1'], Tran_world_forearm)

    xpos['ring3'], xquat['ring3'], T_ring2_ring3 = forward_kinematics(
        [-0.000635, -0.0245, 0], 
        None,
        angles[:,17], [1., 0, 0], T_ring1_ring2, xquat['ring2'], Tran_world_forearm)

    #=====================================PINKY=============================================
    xpos['pinky0'], xquat['pinky0'], T_palm_pinky0 = forward_kinematics(
        [-0.0562459, -0.0554001, -0.00563858], 
        [0.996195, 0, -0.0871557, 0], 
        angles[:,18], [0., 0, -1], T_world_palm, xquat['palm'], Tran_world_forearm)

    xpos['pinky1'], xquat['pinky1'], T_pinky0_pinky1 = forward_kinematics(
        [6.26e-005, -0.0178999, 0], 
        None,
        angles[:,19], [1., 0, 0], T_palm_pinky0, xquat['pinky0'], Tran_world_forearm)

    xpos['pinky2'], xquat['pinky2'], T_pinky1_pinky2 = forward_kinematics(
        [0.000578, -0.033, 0.0005], 
        None,
        angles[:,20], [1., 0, 0], T_pinky0_pinky1, xquat['pinky1'], Tran_world_forearm)

    xpos['pinky3'], xquat['pinky3'], T_pinky2_pinky3 = forward_kinematics(
        [-4.78e-005, -0.0175, 0], 
        None,
        angles[:,21], [1., 0, 0], T_pinky1_pinky2, xquat['pinky2'], Tran_world_forearm)

    return xpos, xquat

def forward_kinematics(PartPos, PartRot, angle, axis, T_parent, xquat_parent, Tran_world_forearm):
    PartPos1 = tf.tile(tf.reshape(tf.constant(PartPos + [1]), [1, 4, 1]), [angle.shape[0], 1, 1])
    PartPos = tf.tile(tf.reshape(tf.constant(PartPos), [1, 3]), [angle.shape[0], 1])
    if PartRot is None:
        PartRot = tf.eye(4, 4, [angle.shape[0]])
    else:
        PartRot = tf.tile(tf.reshape(tf.constant(quaternion_matrix([PartRot]), dtype=tf.float32), [1, 4, 4]), [angle.shape[0], 1, 1])
    rot = tf.matmul(PartRot, angle_axis_rotation_matrix(angle, axis))
    xquat = tf.matmul(xquat_parent, rot)
    T_child = tf.matmul(T_parent, tf.matmul(translation_matrix(PartPos), rot))
    xpos = tf.matmul(Tran_world_forearm, tf.matmul(T_parent, PartPos1))[:,:3,0]
    return xpos, xquat, T_child