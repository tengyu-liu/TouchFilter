import pickle

import numpy as np
import torch
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
_ = plt.ion()

from CodeUtil import *
from HandModel import HandModel
from Losses import FCLoss
from ObjectModel import ObjectModel
from PenetrationModel import PenetrationModel
from EMA import EMA


basin_labels, basin_minima, basin_minima_energies, item_basin_barriers = pickle.load(open('adelm_7/ADELM_dispatch.pkl', 'rb'))
item_basin_barriers = np.array(item_basin_barriers)
basin_basin_barriers = np.zeros([len(basin_minima), len(basin_minima)]) + np.inf

if os.path.exists('logs/zeyu_5p/final_optim.pkl'):
  obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = pickle.load(open('logs/zeyu_5p/final_optim.pkl', 'rb'))
else:
  obj_code, z, contact_point_indices = [], [], []
  linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = [],[],[],[],[],[]
  for _id in range(8):
    fn = 'logs/zeyu_5p/final_%d_optim.pkl'%_id
    _obj_code, _z, _contact_point_indices = pickle.load(open(fn, 'rb'))
    fltr = []
    for i in range(len(_z)):
      print('\r%d:%d'%(_id,i), end='')
      # print(_contact_point_indices[i])
      _linear_independence, _force_closure, _surface_distance, _penetration, _z_norm, _normal_alignment = compute_energy(_obj_code[[i]], _z[[i]], _contact_point_indices[[i]], sd_weight=1, verbose=True)
      if _force_closure.sum() < 1e-5 and _linear_independence < 1e-5 and _surface_distance.sum() < 0.01 and _penetration.sum() < 0.01:
        obj_code.append(_obj_code[i].detach())
        z.append(_z[i].detach())
        contact_point_indices.append(_contact_point_indices[i].detach())
        linear_independence.append(_linear_independence[0].detach())
        force_closure.append(_force_closure[0].detach())
        surface_distance.append(_surface_distance[0].detach())
        penetration.append(_penetration[0].detach())
        z_norm.append(_z_norm[0].detach())
        normal_alignment.append(_normal_alignment[0].detach())
  obj_code = torch.stack(obj_code, dim=0)
  z = torch.stack(z, dim=0)
  contact_point_indices = torch.stack(contact_point_indices, dim=0)
  pickle.dump([obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment], open('logs/zeyu_5p/final_optim.pkl', 'wb'))

print(len(obj_code), len(basin_labels))
