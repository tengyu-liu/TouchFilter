# Progress: 
with_noise,0.1x90 works very well

## Parameters:
* batch_size            8
* name                  with_noise,0.1x90
* restore_epoch         3  
* restore_batch         3500
* epochs                100
* langevin_steps        90
* step_size             0.1
* situation_invariant   False
* adaptive_langevin     True
* clip_norm_langevin    True
* two_stage_optim       -1
* debug                 False
* d_lr                  1e-3
* beta1                 0.99
* beta2                 0.999
* tb_render             True

# Next step
* hand-z prior as a weighted sum of joint angles
* Transfer across different hand shapes - is it possible?

# TODO (from Dr. Zhu):
1. Visualize \phi and \psi: contact point coloring, distance to object, etc.
2. Visualize energy landscape and show grasp types
    * randomly sample points and collect energy
    * plot with TSNE