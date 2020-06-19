import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, n_latent_factor, bn_decay=None, weight_decay=0.0):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay, 
                         weight_decay=weight_decay)
    # net = tf_util.conv2d(net, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=False, is_training=is_training,
    #                      scope='conv2', bn_decay=bn_decay, 
    #                      weight_decay=weight_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64, weight_decay=weight_decay)

    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay, 
                         weight_decay=weight_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay, 
                         weight_decay=weight_decay)
    # net = tf_util.conv2d(net, 1024, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=False, is_training=is_training,
    #                      scope='conv5', bn_decay=bn_decay, 
    #                      weight_decay=weight_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    # net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
    #                               scope='fc1', bn_decay=bn_decay, 
    #                               weight_decay=weight_decay)
    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, 
                                  weight_decay=weight_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, n_latent_factor, activation_fn=tf.nn.sigmoid, scope='fc3', 
                                  weight_decay=weight_decay)

    return net


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.scalar_summary('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.scalar_summary('mat_loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)