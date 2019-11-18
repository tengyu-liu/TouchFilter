import os

import trimesh as tm
import numpy as np

for cup_id in range(1,11):
    cup = tm.load(os.path.join(os.path.dirname(__file__), '../onepiece/%d.obj'%cup_id))
    
    pts = np.random.random((10000, 3)) * 0.2 - 0.1
    pts = np.vstack([pts, cup.sample(10000)])
    dists = tm.proximity.signed_distance(cup, pts)

    pts = pts[np.logical_not(np.isnan(dists)),:]
    dists = dists[np.logical_not(np.isnan(dists))]

    step_size = 0.1
    factor = 1

    step = 0
    while len(dists) < 500000:
        step += 1
        # print '%d'%step
        next_step = pts + np.random.random(pts.shape) * step_size - step_size / 2
        new_distance = tm.proximity.signed_distance(cup, next_step)
        # print 'avg_sq_dist: ', np.mean(np.square(new_distance))
        acceptance = np.exp(np.minimum(np.power(dists, 2) - np.power(new_distance, 2), 0) * factor)
        discard = np.logical_or(acceptance < np.random.random(acceptance.shape), np.isnan(new_distance))
        next_step[discard] = pts[discard]
        new_distance[discard] = dists[discard]
        # print(discard.sum())
        pts = np.vstack([pts, next_step[np.logical_not(discard), :]])
        dists = np.concatenate([dists, new_distance[np.logical_not(discard)]])

        print('\r[%d] %d, %d'%(cup_id, sum(dists > 0), sum(dists < 0)), end='', flush=True)

    shuffle_idx = np.random.permutation(np.arange(len(dists)))
    pts = pts[shuffle_idx,:]
    dists = dists[shuffle_idx]

    np.save('cup_%d.npy'%cup_id, np.concatenate([pts.reshape([-1,3]), dists.reshape([-1,1])], axis=-1))