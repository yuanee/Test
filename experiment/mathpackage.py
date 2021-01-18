#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:18:46 2018

@author: yuanneu
"""


import numpy as np
import scipy as sc
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score


def logis(x,y,beta): #logistic function
    c=-y*(np.dot(x,beta))
    if -35<=c<=35:
       b=1.0/(1+math.exp(c))
    else:
        if c>35:
            b=1.0/(1+math.exp(35))
        else:
            b=1.0/(1+math.exp(-35))    
    return b

def shermanMorissonUpdate(Ainv,x):
    z1 = np.dot(Ainv,x)
    return Ainv - np.outer(z1,z1)/(1+np.dot(x,z1))

def Hp(p): # generate a probability entropy H(p)
    b=[p[0],1-p[0]]
    for i in p[1::]:
        b=[i*k for k in b]+[(1-i)*k for k in b]
    I=sc.stats.entropy(b, qk=None, base=None)
    return I

def lamfunction(x):
    if -20<=x<=20:
        z=(math.exp(x)-1)/(4*x*(math.exp(x)+1))
    else:
        if x<=-20:
            z=-1/(4*x)
        else:
            z=1/(4*x)
    return z

def EMupdateVariational(mu0,Sigma0,Xab,Yab):## the prior distribution N(mu0,sigma0),the initial value xi0
# the given absolute feature Xab is NXd, Yab is the given label{0,+1}.
    length=len(Yab)
    xi=np.ones(length)/2
    Value=1
    invS0=np.linalg.inv(Sigma0)
    Featurebiase=0
    for unit in range(length):
        Featurebiase+=0.5*Yab[unit]*Xab[unit,:]
    Sigma=1*Sigma0
    Value=True
    while(Value):
        Mapmu0=np.dot(invS0,mu0)
        mu=np.dot(Sigma,Mapmu0+Featurebiase)
        invS=invS0.copy()
        for unit in range(length):
            invS+=2*lamfunction(xi[unit])*np.outer(Xab[unit,:],Xab[unit,:])
        Sigma=np.linalg.inv(invS)
        SigmaPxdot=Sigma+np.outer(mu,mu)
        xi_old=xi.copy()
        for unit in range(length):
            xi[unit]=np.sqrt(np.dot(Xab[unit,:],np.dot(SigmaPxdot,Xab[unit,:])))
        if np.linalg.norm((xi-xi_old))<=1e-14:
            Value=False
        else:
            pass
    return mu,Sigma
   

        
 
