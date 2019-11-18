
# TODO: 
1. Setup 1-dim joint angles and joint-angle-priors. This will also solve p1 since hopefully the network will learn that points on the inner surface of a curvy shape are contact points     [check]
2. Add another option as repulsive points (maybe substitute the non-contact points), and possibly make it parameterized, such that the parameter controls in what radius are they repulsive
3. Add part ids + which side it is as additional features for each point.   [check]
4. Multi-grid optimization: first do global parameters, then local ones     [check]

# Problems:
1. PointNet has no knowledge of point sequencing, which means there is no point-prior on which ones tend to be contact points
2. Joint angles don't need to have 360 deg freedom, therefore don't need 2-dim representation. 1-dim is good enough. This will make hand-prior trivial

TODO: 
0. Visualize latest result from with_noise,0.1x90
1. Run with_noise,0.1x90 without noise, see how it performs
