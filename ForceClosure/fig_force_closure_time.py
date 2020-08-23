import time
import torch
import numpy as np
from Losses import *
from ObjectModel import ObjectModel
from CodeUtil import *

code, idx = get_obj_code_random(1)
obj_model = ObjectModel()
fc_loss = FCLoss(obj_model)

for n_cp in [3,5,10,20,100,1000]:
    pts = torch.rand([10000, 1, n_cp, 3], device='cuda', requires_grad=True)
    t = []
    p = pts[0]
    dist = obj_model.distance(code, p)
    grad = obj_model.gradient(p, dist, retain_graph=True, allow_unused=True)
    loss = fc_loss.fc_loss(pts[0], grad, code)
    for i in range(len(pts)):
        p = pts[i]
        t0 = time.time()
        dist = obj_model.distance(code, p)
        grad = obj_model.gradient(p, dist, retain_graph=True, allow_unused=True)
        loss = fc_loss.fc_loss(pts[i], grad, code)
        t.append(time.time() - t0)
    print(n_cp, np.mean(t), np.std(t))

import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

x = [3,5,10,20,100,1000]
y = [3.119656, 3.155564, 3.127634, 3.37597, 3.381954, 3.558481]
_ = plt.plot(x,y, linewidth=3, markersize=10)
_ = plt.plot([x[0], x[-1]], [y[0]+0.005, y[-1]+0.005], linestyle='dashed', linewidth=3)
_ = plt.xlabel('# Contact Points', fontsize=20)
_ = plt.ylabel('Time per 1000 FC Calls (s)', fontsize=20)
_ = plt.xscale('log')
_ = plt.show()