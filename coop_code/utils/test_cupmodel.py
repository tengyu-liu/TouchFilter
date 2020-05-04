import numpy as np
import tensorflow as tf
from CupModel import CupModel
import plotly.graph_objects as go

cup_model = CupModel(3)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

pts = np.random.random([100000, 3]) * 0.3
dist, grad = sess.run(cup_model.pred, feed_dict={cup_model.x: pts})
dist = dist[:,0]
cup = pts[dist>0,:]

fig_data = [
    # go.Scatter3d(x=cup[:,0], y=cup[:,1], z=cup[:,2], mode='markers', marker=dict(color=dist[dist>0])), 
    go.Cone(x=cup[:,0], y=cup[:,1], z=cup[:,2], u=grad[dist>0,0], v=grad[dist>0,1], w=grad[dist>0,2])
]

fig = go.Figure(data=fig_data)

fig.show()


"""Conclusion: cup model grad points towards the decision boundary"""
