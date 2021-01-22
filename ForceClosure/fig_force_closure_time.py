import time
import torch
import numpy as np
from Losses import *
from ObjectModel import ObjectModel
from CodeUtil import *

code, idx = get_obj_code_random(1)
obj_model = ObjectModel()
fc_loss = FCLoss(obj_model)

n_cps = [3,5,10,20,100,1000]
times = []

for n_cp in n_cps:
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
    times.append(np.mean(t))
    print(n_cp, np.mean(t))

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

_ = plt.plot(n_cps,times,linewidth=3, markersize=10)
_ = plt.xlabel('# Contact Points', fontsize=20)
_ = plt.ylabel('Time per 1000 FC Calls (s)', fontsize=20)
_ = plt.xscale('log')
_ = plt.show()
