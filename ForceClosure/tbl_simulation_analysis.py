import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

obj_code, z, contact_point_indices, linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = pickle.load(open('logs/rerun/final_optim.pkl', 'rb'))

results = np.load('simulate_result/%f_%d.npy'%(0,1))

linear_independence = torch.stack(linear_independence)
force_closure = torch.stack(force_closure)
surface_distance = torch.stack(surface_distance)
penetration = torch.stack(penetration)
z_norm = torch.stack(z_norm)
normal_alignment = torch.stack(normal_alignment)

energy = (surface_distance).detach().cpu().numpy()

idx = np.argsort(energy)

success_rates = []
for i in range(1, len(idx)+1):
  success_rates.append(results[idx[:i]].mean())

for i in range(100):
  print(i, idx[i], success_rates[i], energy[idx[i]])

def find(th):
  for i in range(len(idx)):
    if energy[idx[i]] > th:
      return success_rates[i-1]
  return success_rates[-1]
    
for th in [0.0002,0.0005,0.0015,0.0025,0.0035,0.0045,0.02]:
  print(th, find(th))

plt.plot(energy[idx], success_rates)
plt.xlabel('|d(x,O)|')
plt.ylabel('success rate')
plt.show()