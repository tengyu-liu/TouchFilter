import os

import trimesh as tm
import numpy as np

for cup_id in range(1,11):
    cup = tm.load(os.path.join(os.path.dirname(__file__), '../onepiece/%d.obj'%cup_id))
    
    pts = np.random.random((1000, 3)) * 0.2 - 0.1
    pts = np.vstack([pts, obj.sample(1000)])
    dists = tm.proximity.signed_distance(obj, pts)

    pts = pts[np.logical_not(np.isnan(dists)),:]
    dists = dists[np.logical_not(np.isnan(dists))]

    xs = np.copy(pts)
    ys = np.copy(dists)

    step = 0
    while sum(ys > 0) < 10000 or sum(ys < 0) < 10000:
        step += 1
        # print '%d'%step
        next_step = pts + np.random.random(pts.shape) * step_size - step_size / 2
        new_distance = tm.proximity.signed_distance(obj, next_step)
        # print 'avg_sq_dist: ', np.mean(np.square(new_distance))
        acceptance = np.exp(np.minimum(np.power(dists, 2) - np.power(new_distance, 2), 0) * factor)
        discard = np.logical_or(acceptance < np.random.random(acceptance.shape), np.isnan(new_distance))
        next_step[discard] = pts[discard]
        new_distance[discard] = dists[discard]
        # print(discard.sum())
        xs = np.vstack([xs, next_step[np.logical_not(discard), :]])
        ys = np.concatenate([ys, new_distance[np.logical_not(discard)]])
        pts = next_step
        dists = new_distance

        sys.stdout.write('\r[%s] %d, %d'%(part, sum(ys > 0), sum(ys < 0)))
        sys.stdout.flush()

    shuffle_idx = np.random.permutation(np.arange(len(ys)))
    xs = xs[shuffle_idx,:]
    ys = ys[shuffle_idx]

    pos_x = xs[ys >= 0, :]
    pos_y = ys[ys >= 0]

    neg_x = xs[ys < 0, :]
    neg_y = ys[ys < 0]

    pos_x = pos_x[:10000]
    neg_x = neg_x[:10000]
    pos_y = pos_y[:10000]
    neg_y = neg_y[:10000]

    xs = np.vstack([pos_x, neg_x])
    ys = np.concatenate([pos_y, neg_y])
