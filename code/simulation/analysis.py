import matplotlib.pyplot as plt
import numpy as np
import pickle

xs = np.concatenate([np.load('xs_%d.npy'%i) for i in range(6)]).reshape([-1])
es = pickle.load(open('C:/Users/24jas/Desktop/TouchFilter/code/evaluate/synthesis/individual_z2/dynamic_z2_nobn_unitz2/0099-300.pkl', 'rb'))['syn_e']
es = es.reshape([-1])
ps = np.load('penetration.npy')

xs = ((xs<0.1) * (ps>-1)).astype(np.float)

print(xs.shape, es.shape)

plt.scatter(es, xs, s=1)
plt.show()

e_for_plot = np.linspace(1.5,20,10000)
x_for_plot = np.zeros_like(e_for_plot)

for i, e in enumerate(e_for_plot):
    x_for_plot[i] = np.mean(xs[np.argsort(np.abs(es - e))[:500]])

plt.plot(e_for_plot, x_for_plot)
plt.xlabel('energy')
plt.ylabel('success rate')
plt.show()



ax = plt.subplot(221)
ax.set_title('Dist of X (E < 3)')
ax.hist(xs[es < 2.5].astype(np.float), bins=100)
ax = plt.subplot(222)
ax.set_title('Dist of X (E >= 3)')
ax.hist(xs[es >= 2.5].astype(np.float), bins=100)
ax = plt.subplot(223)
ax.set_title('Dist of E (X < 0.1)')
ax.hist(es[xs < 0.1].astype(np.float), bins=100)
ax = plt.subplot(224)
ax.set_title('Dist of E (X >= 0.1)')
ax.hist(es[xs >= 0.1].astype(np.float), bins=100)
plt.show()


