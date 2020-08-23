import plotly
import plotly.graph_objects as go

from Losses import *
from ObjectModel import *
from CodeUtil import *
import torch

n_contact_points = 5
ball_radius = 1
stepsize = 0.01
batch_size = 32

object_model = ObjectModel()

while True:
    code, idx = get_obj_code_random(batch_size)
    loss = FCLoss(object_model)

    pts = torch.rand([batch_size, n_contact_points, 3], device='cuda')
    pts -= pts.mean(dim=1, keepdim=True)
    pts /= torch.norm(pts, dim=-1, keepdim=True)
    pts.requires_grad_()


    for step in range(10000):
        T = 0.98 ** step
        dist = object_model.distance(code, pts)
        grad = object_model.gradient(pts, dist, retain_graph=True, create_graph=True)
        linear_independency, force_closure = loss.fc_loss(pts, grad, code)
        grad = torch.autograd.grad((linear_independency.sum() + force_closure.sum() + torch.square(dist).sum()), pts, retain_graph=True)[0]
        pts2 = pts - 0.5 * grad * stepsize * stepsize + torch.normal(0,1,pts.shape, device='cuda') * stepsize
        dist2 = object_model.distance(code, pts2)
        grad2 = object_model.gradient(pts2, dist2, retain_graph=True, create_graph=True)
        linear_independency2, force_closure2 = loss.fc_loss(pts2, grad2, code)
        alpha = torch.rand(batch_size, device='cuda')
        accept = alpha < torch.exp((linear_independency+force_closure+torch.square(dist).sum(1).squeeze() - 
                    (linear_independency2+force_closure2+torch.square(dist2).sum(1).squeeze()))/T)
        accept = accept.unsqueeze(1).unsqueeze(1)
        pts = pts * (~accept) + pts2 * accept

    dist = object_model.distance(code, pts)
    grad = object_model.gradient(pts, dist)
    grad /= torch.norm(grad, dim=-1, keepdim=True)
    grad = grad.detach().cpu().numpy()
    pts = pts.detach().cpu().numpy()

    for i in range(batch_size):
        print(linear_independency[i].sum().item(), force_closure[i].sum().item(), torch.square(dist[i]).sum().item())
        data = []
        mesh = get_obj_mesh(idx[i])
        data.append(go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2], i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2], color='lightblue'))

        data.append(go.Cone(x=pts[i,:,0], y=pts[i,:,1], z=pts[i,:,2], u=-grad[i,:,0], v=-grad[i,:,1], w=-grad[i,:,2], anchor='tip',
                        colorscale=[(0,'lightpink'), (1,'lightpink')], sizemode='absolute', sizeref=0.2, opacity=0.5))

        fig = go.Figure(data=data)
        fig.update_layout(dict(scene=dict(aspectmode='data')), showlegend=False)
        fig.show()
        input()