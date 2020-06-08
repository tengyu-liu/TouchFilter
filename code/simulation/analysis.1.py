import matplotlib.pyplot as plt
import numpy as np
import pickle

# xs = np.concatenate([np.load('xs_%d.npy'%i) for i in range(6)]).reshape([-1])
xs = np.load('xs.npy')
es = pickle.load(open('../evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))['syn_e']
es = es.reshape([-1])
ps = np.load('penetration.npy')

result = []
for i in range(1000):
  candidate = -1
  count = 0
  while count < 10:
    idx = np.random.randint(0, len(xs)-1)
    if es[idx] < 10:
      if candidate < 0 or es[idx] < es[candidate]:
        candidate = idx
      count += 1
  
  print('\r', i, candidate, end='', flush=True)
  result.append(xs[candidate])

print()
print(sum(result) / len(result))

print(sum(xs) / len(xs))

# best result: 77%
