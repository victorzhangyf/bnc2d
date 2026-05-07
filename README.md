# bnc2d

Compiling class formulas of multi-class BNCs into OR-Decomposable NNFs. 

Zhang Y., Darwiche A. Scaling the Explanation of Multi-Class Bayesian Network Classifiers. To Appear in
The 4th World Conference on Explainable Artificial Intelligence (XAI), 2026


## set up

- Two methods of constructing jointrees can be specified in `global_config.py`, you need at least one of them
    - `c2d`: requires http://reasoning.cs.ucla.edu/c2d/
    - `samiam`: requires java installed on your system
- The arithmetic circuit module is for debug only (computing ground truth marginal for verification), requires `ace` (http://reasoning.cs.ucla.edu/ace/) with `-forceC2d` option



