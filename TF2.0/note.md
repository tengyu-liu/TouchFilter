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
* Pipeline
    * 