import numpy as np
import pickle

xs = np.concatenate([np.load('xs_%d.npy'%i) for i in range(6)]).reshape([-1])
es = pickle.load(open('/media/tengyu/BC9C613B9C60F0F6/Users/24jas/Desktop/TouchFilter/code/evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))['syn_e']
es = es.reshape([-1])

import matplotlib.pyplot as plt
plt.scatter(es, xs, s=1)
plt.show()


ax = plt.subplot(221)
ax.set_title('Dist of X (E < 3)')
ax.hist(xs[es < 2.5], bins=100)
ax = plt.subplot(222)
ax.set_title('Dist of X (E >= 3)')
ax.hist(xs[es >= 2.5], bins=100)
ax = plt.subplot(223)
ax.set_title('Dist of E (X < 0.1)')
ax.hist(es[xs < 0.1], bins=100)
ax = plt.subplot(224)
ax.set_title('Dist of E (X >= 0.1)')
ax.hist(es[xs >= 0.1], bins=100)
plt.show()


