# Plan
* We choose PCA size = 2 according to the eigen-grasp paper
* D(h,o) = \sum_{p\in h} (1 - W) * f_0(p,o) + W * f_1(p,o) + ||W||
    * where f_0 is quadratic function and f_1 is quadratic with negative half zero'd
* W can be either a function of current situation, which is [f_0(h, o), h], or invariant to current situation
* If W is variant to situation:
    * W = f(f_0(h,o), h)
    * parameter of f is learned
* If W is invariant to situation:
    * W is learned


# Experiment
* 2019-10-21
    * bear
        * CUDA_VISIBLE_DEVICES=0 python train.py --name exp1 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-3
        * CUDA_VISIBLE_DEVICES=1 python train.py --name exp2 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-3 --situation_invariant
    * camel
        * CUDA_VISIBLE_DEVICES=0 python train.py --name exp3 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-2
        * CUDA_VISIBLE_DEVICES=1 python train.py --name exp4 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-2 --situation_invariant
        * CUDA_VISIBLE_DEVICES=2 python train.py --name exp5 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-4
        * CUDA_VISIBLE_DEVICES=3 python train.py --name exp6 --batch_size 32 --use_pca --z_size 2 --pca_size 44 --adaptive_langevin --clip_norm_langevin --penalty_strength 1e-4 --situation_invariant
