import os
import sys
import time
import datetime
import numpy as np
import trimesh as tm
import scipy.io as sio
import tensorflow as tf

from model_inference import PartModel, CupModel

from pyquaternion import Quaternion as Q

base_path = '../../'
size = 64

cup_model_dir = os.path.join(base_path, 'cups/onepiece')

plot = False
part_weight_path = os.path.join(base_path, 'GloveSDF', 'models')
cup_weight_path = os.path.join(base_path, 'cup_models_reorder')
data_path = os.path.join(base_path, 'MujocoSDF/data/grasp')
output_path = os.path.join('..', 'grasp_sdf')

if not os.path.exists(output_path):
    os.mkdir(output_path)

# Generate grid coordinates
axis = np.linspace(-0.3, 0.3, size)
pts = np.transpose(np.stack(np.meshgrid(axis, axis, axis), axis=-1), [1,0,2,3])
pts = pts.reshape([-1,3])

# Load part models
parts = ['palm',
            'thumb0', 'thumb1', 'thumb2', 'thumb3',
            'index0', 'index1', 'index2', 'index3',
            'middle0', 'middle1', 'middle2', 'middle3',
            'ring0', 'ring1', 'ring2', 'ring3',
            'pinky0', 'pinky1', 'pinky2', 'pinky3']

part_models = {p: PartModel(p, 3999, part_weight_path) for p in parts}
cup_models = {cup_id: CupModel(cup_id, 999, cup_weight_path) for cup_id in range(1,11)}

c = tf.ConfigProto()
c.gpu_options.allow_growth = True

sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

t0 = time.time()
finished = 0

if plot:
    import mayavi.mlab as mlab

cup_annotation = {x.split(',')[0]: x.strip().split(',')[1:] for x in open(os.path.join(os.path.dirname(__file__), '../../GraspingDeepFRAME/data/cup_video_annotation.txt')).readlines()}

cup_id = int(sys.argv[1])


v = tm.load_mesh(os.path.join(cup_model_dir, '%d.obj'%cup_id))
center = np.mean(v.bounding_box.vertices, axis=0)

total = 0
for data_id in range(1,11):
    for pair in cup_annotation['%d_%d'%(cup_id, data_id)]:
        start, end = [int(x) for x in pair.split(':')]
        total += end - start

for data_id in range(1,11):
    
    # Load hand configuration
    glove_data = sio.loadmat(os.path.join(data_path, 'cup%d'%cup_id, 'cup%d_grasping_60s_%d.mat'%(cup_id, data_id)))['glove_data']

    for pair in cup_annotation['%d_%d'%(cup_id, data_id)]:
        start, end = [int(x) for x in pair.split(':')]
        for frame in range(start-1, end):
            # if os.path.exists(os.path.join(output_path, 'cup_%d_grasp_%d_frame_%d.npy'%(cup_id, data_id, frame))):
            #     continue

            sdf_h = None
            
            # Compute InteractionField
        
            # hand_center = np.mean([glove_data[frame, 1 + (6 + i) * 3 : 1 + (7 + i) * 3] - glove_data[frame, 1 + 27 * 3: 1 + 28 * 3] - Q(glove_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotate(center) for i in range(21)], axis=0)
            hand_center = glove_data[frame, 1 + 6 * 3 : 1 + 7 * 3]- glove_data[frame, 1 + 27 * 3: 1 + 28 * 3] - Q(glove_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4]).rotate(center)
            hand_center /= np.linalg.norm(hand_center)

            v1 = np.array([1.0, 0.0, 0.0])
            q = np.array([1.0, 0.0, 0.0, 0.0])
            
            if np.linalg.norm(hand_center - v1) > 1e-15:
                q[1:] = np.cross(v1, hand_center)
                q[0] = 1 + np.dot(v1, hand_center)
                q /= np.linalg.norm(q)
            
            q = Q(glove_data[frame, 1 + 28 * 3 + 6 * 4 : 1 + 28 * 3 +  7 * 4]).inverse
            # q = Q(q)

            t = [q.inverse.rotate(glove_data[frame, 1 + (6 + i) * 3 : 1 + (7 + i) * 3] - glove_data[frame, 1 + 27 * 3: 1 + 28 * 3] - (Q(glove_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4])).rotate(center)) for i in range(21)]
            r = [q.inverse * Q(glove_data[frame, 1 + 28 * 3 + (6 + i) * 4: 1 + 28 * 3 + (7 + i) * 4]) for i in range(21)]

            part_pts  = [np.matmul(r[i].inverse.rotation_matrix, (pts * 0.8 - t[i]).T).T for i in range(21)]
        
            sdf_h = np.array(sess.run([part_models[p].pred for p in parts], feed_dict={part_models[p].x: part_pts[parts.index(p)] for p in parts})).reshape([21, size, size, size])
            sdf_h = np.max(sdf_h, axis=0)

            r = Q(glove_data[frame, 1 + 28 * 3 + 27 * 4: 1 + 28 * 3 + 28 * 4])
            
            # cup_pts = np.matmul(Q(r).inverse.rotation_matrix, (np.matmul(q.inverse.rotation_matrix, pts.T).T + Q(r).rotate(center)).T).T * 2
            cup_pts = np.matmul((Q(r).inverse * q).rotation_matrix, pts.T).T * 4 + center * 5
            if cup_id == 9:
                cup_pts = np.matmul((Q(r).inverse * q).rotation_matrix, pts.T).T * 2 + center * 5
            
            sdf_c = sess.run(cup_models[cup_id].pred, feed_dict={cup_models[cup_id].x: cup_pts}).reshape([size, size, size])

            xc, yc, zc = np.where(sdf_c > -0.01)
            xh, yh, zh = np.where(sdf_h > -0.001)

            if plot:
                mlab.clf()
                half = int(size/2)
                mlab.points3d([0,0,0,0,size,size,size,size], [0,0,size,size,0,0,size,size], [0,size,0,size,0,size,0,size], mode='point', opacity=0.0)
                mlab.points3d(np.arange(half, size), np.zeros([half]) + half, np.zeros([half]) + half, mode='point', color=(1, 1, 1))
                mlab.points3d(np.zeros([half]) + half, np.arange(half, size), np.zeros([half]) + half, mode='point', color=(1, 1, 1))
                mlab.points3d(np.zeros([half]) + half, np.zeros([half]) + half, np.arange(half, size), mode='point', color=(1, 1, 1))
                mlab.points3d(xc, yc, zc,
                                    mode="cube",
                                    color=(0, 0, 1),
                                    scale_factor=1)
                mlab.points3d(xh, yh, zh,
                                    mode="cube",
                                    color=(1, 0, 0),
                                    scale_factor=1)
                # mlab.points3d(xh2, yh2, zh2,
                #                     mode="cube",
                #                     color=(0, 1, 0),
                #                     scale_factor=1)
                mlab.savefig('/dev/null/1.jpg')

            np.save(os.path.join(output_path, 'cup_%d_grasp_%d_frame_%d.npy'%(cup_id, data_id, frame)), np.stack([sdf_c, sdf_h], axis=-1))
            finished += 1
            elapsed = time.time() - t0
            ETA = elapsed / finished * (total - finished)
            print('\rCup %d Grasp %d Frame %d (%d/%d) ETA: %s'%(cup_id, data_id, frame, finished, total, datetime.timedelta(seconds=ETA)))

count = 0
all_sdf = []
for grasp_id in range(1,11):
    for pair in cup_annotation['%d_%d'%(cup_id, grasp_id)]:
        start, end = [int(x) for x in pair.split(':')]
        for frame in range(start-1, end):
            count += 1
            print('\r%d/%d'%(count, total), end='')
            sdf = np.load('../grasp_sdf/cup_%d_grasp_%d_frame_%d.npy'%(cup_id, grasp_id, frame)).reshape([size, size, size, 2])
            all_sdf.append(sdf)

all_sdf = np.array(all_sdf)
np.save('../grasp_sdf/grasp_sdf_cup%d_hd.npy'%cup_id, all_sdf)
