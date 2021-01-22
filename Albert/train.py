from model import Vertex2MANO
import torch
import numpy as np

import smplx

# set seed to control randomness
np.random.seed(0)
torch.manual_seed(0)

### hyper parameters ###
batch_size = 128
z_size = 12
hidden_size = 1024
learning_rate = 1e-6
total_steps = int(1e8)
log_interval = 100
viz = True
viz_interval = 1000

if viz:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection='3d')
    loss_history = []

mano = smplx.create(model_path="./models/MANO_RIGHT.pkl", model_type="mano",flat_hand_mean=True,batch_size=batch_size,use_pca=False).cuda()
model = Vertex2MANO(z_size=z_size, hidden_size=hidden_size).cuda()
optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=1e-3)

for step in range(total_steps):
    optimizer.zero_grad()
    z = torch.normal(0,1,[batch_size, z_size], requires_grad=True).cuda().float()
    vert_pred, mano_parameters = model(z)
    mano_vertices = mano(hand_pose=mano_parameters).vertices.detach()
    loss = torch.norm(vert_pred - mano_vertices, dim=-1).mean()# + torch.abs(mano_parameters).mean() * 1e-3
    loss.backward()
    optimizer.step()
    if step % log_interval == 0:
        print('\rstep %d - loss: %.4f'%(step, loss.detach().cpu().numpy()), end='')
        loss_history.append(loss.detach().cpu().numpy())
        ax1.cla()
        ax1.plot(loss_history)
    if viz and step % viz_interval == 0:
        # print()
        # print(mano_parameters[0])
        ax2.cla()
        v = vert_pred[0].detach().cpu().numpy()
        m = mano_vertices[0].detach().cpu().numpy()
        ax2.scatter(v[:,1], v[:,1], v[:,2], s=1, c='red')
        ax2.scatter(m[:,1], m[:,1], m[:,2], s=1, c='blue')
        ax2.set_xlim([-0.2, 0.2])
        ax2.set_ylim([-0.2, 0.2])
        ax2.set_zlim([-0.2, 0.2])
        plt.pause(1e-5)
