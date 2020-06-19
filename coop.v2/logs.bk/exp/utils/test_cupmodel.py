import os
import numpy as np
import tensorflow as tf
from CupModel import CupModel
import plotly.graph_objects as go
import trimesh as tm

for obj_id in [3]:
    pts = np.random.random([100000, 3]) * 0.5 - 0.25
    cup_model = CupModel(obj_id)
    cup_pred = cup_model.predict(tf.constant(pts, dtype=tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(obj_id)
    cup_mesh = tm.load(os.path.join(os.path.dirname(__file__), '../../data/cups/onepiece/%d.obj'%obj_id))

    dist, grad = sess.run(cup_pred)
    dist = dist[:,0]

    # cup = pts[np.logical_and(dist>-0.005, dist < 0),:]
    # color = dist[np.logical_and(dist>-0.005, dist < 0)]
    cup = pts[dist < 0,:]
    color = dist[dist < 0]
    color -= color.min()
    color /= color.max()

    fig_data = [
        go.Scatter3d(x=cup[:,0], y=cup[:,1], z=cup[:,2], mode='markers', marker=dict(size=2, color=color)), 
        go.Mesh3d(x=cup_mesh.vertices[:,0], y=cup_mesh.vertices[:,1], z=cup_mesh.vertices[:,2], \
                i=cup_mesh.faces[:,0], j=cup_mesh.faces[:,1], k=cup_mesh.faces[:,2], color='lightpink'), 
        # go.Cone(x=cup[:,0], y=cup[:,1], z=cup[:,2], u=grad[dist>0,0], v=grad[dist>0,1], w=grad[dist>0,2])
    ]

    fig = go.Figure(data=fig_data)

    fig.show()

    input()

    """Conclusion: cup model grad points towards the decision boundary"""
