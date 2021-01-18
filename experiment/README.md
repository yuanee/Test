Experimental Design under the Bradley-Terry Model
==========

The code in this repository provides a framework for experimental design algorithm for pairwise data. In particular, the algorithms implemented are described in the paper 

>Experimental Design Under the Bradley-Terry Model.
>Yuan Guo, Peng Tian, Jayashree Kalpathy-Cramer, Susan Ostmo,
J.Peter Campbell, Michael F.Chiang, Deniz Erdo˘gmu¸s, Jennifer Dy and Stratis Ioannidis.
>(IJCAI 2018)

The framework can be optimized by greedy algorithm for the monotone submodular function:
  >Maximize f(S)
  >subj. to S\subseteq C, |S|=K.

## mathpackage.py ##
The python file includes the following modules:
```
numpy
scipy
random
sklearn
```
##### Function `logis`: #####
Logistic function for feature vector x, label y, parameter \beta.

##### Function `shermanMorissonUpdate`: #####
Sherman Morrison formula for A^{-1}

##### Function `EMupdateVariational`: #####
The variational bayesian for logistic regression with prior Gaussian distribution with mean mu0 and covariance Sigma0.
The observation labels are Yab related to the feature Xab. The function will return the approximated posterior Gaussian distribution.

## ActiveLearning ##

This file will return objective S for different experimental design algorithms. The input are:

```
(Xarray,Omega,Hyperparameter,Xabs,Yabs) 
```

* `Xarray` is the feature matrix for N absolute samples.

* `Omega` is the set of possible comparisons.

* `Hyperparameter` is a hyperparameter for each algorithm.

* `Xabs` is the feature matrix for absolute samples in \mathcal{A}.

* `Yabs` is label matrix for absolute samples in \mathcal{A}.

The algorithms are:

```
(CovGreedy,EntGreedy,MutGreedy,FishGreedy) 
```

* `CovGreedy:` Covariance objective function.

* `EntGreedy:` Information Entropy objective function.

* `MutGreedy:` Mutual  Information objective function.

* `FishGreedy:` Fisher Information.

#####  `Fisher Information`: #####

We use the equation (8) in paper:
>Batch mode active learning and its application to medical image classification
>Steven C. H. Hoi, Rong Jin, Jianke Zhu, Michael R. Lyu.
>(ICML 2006)

## Citing This Paper ## 

Please cite the following paper if you intend to use this code for your research.
>Experimental Design Under the Bradley-Terry Model.
>Yuan Guo, Peng Tian, Jayashree Kalpathy-Cramer, Susan Ostmo,
J.Peter Campbell, Michael F.Chiang, Deniz Erdo˘gmu¸s, Jennifer Dy and Stratis Ioannidis.
>(IJCAI 2018)


## Acknowledgement

Our work is supported by NIH (R01EY019474, P30EY10572), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).

