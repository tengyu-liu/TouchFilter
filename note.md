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

# 2020-05-30
## TODO
1. Test that for each epoch, what is the best sampling strategy? How to determine the best sampling strategy
  1. Visualize energy changes at different step size / noise size
  2. find a pattern between step size / noise size
2. find out the best way of scheduling regularization
  1. current problem is that when regularization is small, it cannot regularize the energy explosion. when the regularization is big, the network shrinks to near zero
  2. why does network shrink to near zero? 
    1. when the network is too small, the langevin step becomes too big so it no longer produces positive improvement. 
    2. then, there is nothing for the descriptor to learn. 
3. find out why the generator falls in mode collapse
  1. in the case of energy explosion, the stepsize does not grow with energy growth, therefore not able to produce enough improvement due to too small update steps
  2. in the case of energy deminishing, the stepsize does not shrink with the energy shrinkage, therefore not able to navigate through energy landscape due to too large update steps
3. What if we don't MCMC? Just generate it.