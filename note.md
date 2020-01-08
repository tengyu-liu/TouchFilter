# 2020-01-08
## Notes
1. static-physical and dynamic can synthesize multiple grasping type
2. all experiments show unstable results, but they are easily distinguishable (result energy > 3)
3. naive interpolation between z2 show unstatisfactory results
## TODO
1. Collect different grasp types for illustration (report/figs/types/%d.png)
2. Run t-SNE to show relationship between z2 and grasp type (report/figs/tsne/%s.png)
   1. Need a way to tell different grasp types. Consider clustering on weights
   2. Only keep results with similar energy (for example, around 2)
   3. Maybe consider run this with the same initial condition
   4. Show a few examples in each cluster
   5. X,Y: t-SNE coordinates. Z: energy. color: cluster
3. Show path between points. Consider magetic diffusion method. (report/figs/interpolation/%s.png)
   1. Pick two points A and B
   2. run MCMC to minimize [E(x) + d(x,B)] starting from A

