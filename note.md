# 2019-10-01
## Current progress/problem:
* Touch energy and prior energy explodes in opposite directions, cancelling out each other but makes training unstable
    * With 10x L2 regularization on both D and G, the energy no longer explodes but retracts instead to very small numbers. Possibly due to generator produce too small result each time.
    * With 10x L2 on D and smaller L2 on G, the energy still explodes. 
        * It seems like that 10x L2 on D and smaller L2 on G is stablizing towards the end of epoch 2. However, they still suffer from instability. 
    * **Proposed Solution** Try prior multiplier
    * **Proposed Solution** Try sigmoid energy
* Current experiments:
    * On camel: 
        * C1F20DW10GW10: 
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 10x L2 norm
        * C1F20DW3GW-2
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm    <!-- this is a typo in exp name. descriptor L2 is 10x instead of 3x -->
            * generator is regularized with 0.01x L2 norm
        * C1F20DW10GW-1
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 0.1x L2 norm
        * C1F20DW10GW1
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 1x L2 norm
    * On bear: 
        * C1F20DW1GW1P-1
            * 1 channel
            * 20 filters
            * descriptor is regularized with 1x L2 norm
            * generator is regularized with 1x L2 norm
            * prior energy has a multiplier 0.1
        * C1F20DW1GW1P-2
            * 1 channel
            * 20 filters
            * descriptor is regularized with 1x L2 norm
            * generator is regularized with 1x L2 norm
            * prior energy has a multiplier 0.01
        * C1F20DW1GW1P-3
            * 1 channel
            * 20 filters
            * descriptor is regularized with 1x L2 norm
            * generator is regularized with 1x L2 norm
            * prior energy has a multiplier 0.001
        * C1F20DW1GW1P0
            * 1 channel
            * 20 filters
            * descriptor is regularized with 1x L2 norm
            * generator is regularized with 1x L2 norm
            * prior energy has a multiplier 1
    <!-- * On donkey: C1F20DW10GW10S, C1F20DW3GW-2S, C1F20DW10GW-1S, C1F20DW10GW1S -->

# 2019-10-03
## Current Progress/Problem
* With smaller G-loss weight, at about 19.85k ~ 21.08k steps, all indicators take mysterious sharp turns and start to stablize. 
    * Langevin step mainly focuses on the last 9 channels (gpos + grot). 
        * **Proposed Solution** Keep track of rolling mean of gradient, and normalize by that. 
* With prior multiplier, synthesized prior energy diminishes.
    * For all experiments, we haven't achieved to have synthesized Z to have similar scale as GT Z. This is probably because of the inbalance in gradient distribution, however we are not sure of that yet.
* Current experiments:
    * On bear: continue C1F20DW1GW1P-1, C1F20DW1GW1P-2, C1F20DW1GW1P-3, C1F20DW1GW1P0
    * On camel: 
        * C1F20DW10GW-2MG
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 0.01x L2 norm
            * langevin update adapts to mean of gradients
        * C1F20DW10GW-1MG
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 0.1x L2 norm
            * langevin update adapts to mean of gradients
        * C1F20DW10GW1MG
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 1x L2 norm
            * langevin update adapts to mean of gradients
        * C1F20DW10GW-2MG-S: 
            * 1 channel
            * 20 filters
            * descriptor is regularized with 10x L2 norm
            * generator is regularized with 0.01x L2 norm
            * langevin update adapts to mean of gradients
            * energy output is altered by a sigmoid function