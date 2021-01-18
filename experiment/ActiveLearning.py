#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:52:25 2018

@author: yuanneu
"""

from mathpackage import *


class Greedy():
    """ A generic abstract class implementing the simple greedy algorithm."""
        
    def __init__(self,Xarray,Hyperparameter,X_abs,Y_abs,Omega):
        """Initialization"""
        self.Xarray = Xarray
        self.Omega = Omega
        self.Hyper=Hyperparameter
        self.X_abs=X_abs
        self.Y_abs=Y_abs
        self.n,self.d= np.shape(self.X_abs)

    def initializeS(self):
        """Initialize set S.
        """
        self.S = []
        self.Remainder = self.Omega.copy()
    
    def updateS(self,key):
        """Update current set S with element x.
        """
        self.S.append(key)
        self.Remainder.remove(key)

    def getScopy(self):
        return self.S.copy()

    def getRemainderCopy(self):
        return self.Remainder.copy()
    
    def greedyGrow(self,k):
        """Construct greedy set.
        """
        while len(self.S)<k:
            maxVal = float("-inf")
            for i in self.Remainder:
                val = self.evalSaddX(i)
                if val>maxVal:
                    maxVal=val
                    opti = i
            self.opti=opti
            self.updateS(opti)
        
        return self.getScopy()
    
class CovGreedy(Greedy):
    def initializeS(self):
        Greedy.initializeS(self)
        self.invA0=1/float(self.Hyper)*np.eye(self.d)
        for j in range(self.n):
            self.invA0+=np.outer(self.X_abs[j,:],self.X_abs[j,:])
    def updateS(self,key):
        Greedy.updateS(self,key)
        x_ij=self.Xarray[key[0],:]-self.Xarray[key[1],:]
        self.invA0=shermanMorissonUpdate(self.invA0,x_ij)
        
    def evalSaddX(self,key):
        x = self.Xarray[key[0],:]-self.Xarray[key[1],:]
        z = np.dot(self.invA0,x)
        return np.dot(x,z)

   
class EntGreedy(Greedy):
    def initializeS(self):
        Greedy.initializeS(self)
        self.clf=LogisticRegression(C=self.Hyper,fit_intercept=False, penalty='l2', tol=0.01)
        self.X_abs=np.concatenate((np.zeros((1,self.d)),self.X_abs),0)
        if self.Y_abs[0]==1:   
            self.Y_abs=np.concatenate(([-1],self.Y_abs))
        else:
            self.Y_abs=np.concatenate(([1],self.Y_abs))
        self.clf.fit(self.X_abs,self.Y_abs)
        self.beta=self.clf.coef_[0]
    def evalSaddX(self,key):
        x= self.Xarray[key[0],:]-self.Xarray[key[1],:]
        return -abs(np.dot(self.beta,x))    

    
class MutGreedy(Greedy):
    def initializeS(self):
        Greedy.initializeS(self)
        X_com=np.array([self.Xarray[key[0],:]-self.Xarray[key[1],:] for key in self.Remainder])
        Com_list=[key for key in self.Remainder]
        mu0=np.zeros(self.d)
        Sigma0=self.Hyper*np.eye(self.d)
        mu,Sigma=EMupdateVariational(mu0,Sigma0,self.X_abs,self.Y_abs)
        Num=6000
        BetaM=np.random.multivariate_normal(mu,Sigma,Num)
        self.P=np.ones((1,Num))
        self.G={}
        self.W={}
        for key in self.Remainder:
            x_ij=self.Xarray[key[0],:]-self.Xarray[key[1],:]
            garray=np.array([logis(x_ij,1,BetaM[s,:]) for s in range(Num)])
            w1=np.array([Hp([garray[s]]) for s in range(Num)])
            self.G[key]=garray
            self.W[key]=np.mean(w1)
    def evalSaddX(self,key):
        Z1=np.concatenate((self.P*self.G[key],self.P*(1-self.G[key])),0)
        entropy=sc.stats.entropy(np.mean(Z1,1),qk=None,base=None)
        return entropy-self.W[key]
    def updateS(self,key):
        Greedy.updateS(self,key)
        self.P=np.concatenate((self.P*self.G[key],self.P*(1-self.G[key])),0)


class FishGreedy(Greedy):
    def initializeS(self,delta=0.001):
        Greedy.initializeS(self)
        self.delta=delta
        self.Remainkey=[]
        N_item=len(self.Remainder)
        self.Remainlist=[i for i in range(N_item)]
        Xnorm=[]
        for key in self.Remainder:
            x_ij=self.Xarray[key[0],:]-self.Xarray[key[1],:]
            Xnorm.append(x_ij/np.linalg.norm(x_ij))
            self.Remainkey.append(key)
        self.Xnorm=np.array(Xnorm)
        xt_xp=np.dot(self.Xnorm,self.Xnorm.T)
        self.xsquare=np.multiply(xt_xp,xt_xp)
        clf=LogisticRegression(C=self.Hyper,fit_intercept=False, penalty='l2', tol=0.01) 
        self.X_abs=np.concatenate((np.zeros((1,self.d)),self.X_abs),0)
        if self.Y_abs[0]==1:
            self.Y_abs=np.concatenate(([-1],self.Y_abs))
        else:
            self.Y_abs=np.concatenate(([1],self.Y_abs))
        clf.fit(self.X_abs,self.Y_abs)
        beta=clf.coef_[0]
        self.P=np.zeros(N_item)
        for sam in range(len(self.Remainder)):
            p=logis(self.Xnorm[sam,:],1,beta)
            self.P[sam]=p*(1-p)
        self.gs_Umat=np.zeros(N_item)
        self.gs_UmatX=np.zeros(N_item)
        for j in range(N_item):
            self.gs_Umat[j]=self.P[j]/self.delta
            self.gs_UmatX[j]=1/(1/self.gs_Umat[j]+self.xsquare[j,j])
    def greedyGrow(self,k):
        while len(self.S)<k:
            maxVal = float("-inf")
            for i in self.Remainlist:
                item_minx=list(set(self.Remainlist)-set([i]))
                val=self.gs_Umat[i]+np.dot(self.gs_Umat[item_minx],self.xsquare[i,item_minx])*self.gs_UmatX[i]
                if val>maxVal:
                    maxVal=val
                    opti = i        
            key_opti=self.Remainkey.pop(opti)
            opti_point=self.Remainlist.pop(opti)
            self.S.append(key_opti)
            for j in self.Remainlist:
                self.gs_Umat[j]=1/(1/self.gs_Umat[j]+self.P[opti_point]/self.P[j]*self.xsquare[j,opti_point])
                self.gs_UmatX[j]=1/(1/self.gs_Umat[j]+self.xsquare[j,j])
        return self.getScopy()
            
 

    
