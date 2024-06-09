#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
from models import model

import numpy as np
import matplotlib.pyplot as plt


###########################################
# Classes
###########################################
class CondDiff(model.Model):
    
    def __init__(self, beta = 10, T = 1, n=100, k=1):
        """ Conditioned diffusion model

        Parameters
        ----------
        beta : float
            Beta in SDE.
        T : float
            Total time.
        n : int
            Number of time steps (dt = T/n).
            Note that u in R^n+1 while w in R^n, we will often set N=n+1
        k : int
            Observation at every k time step.
        R : array(m,m) or float
            Observation covariance 
        bg: array(d)
            Prior mean / background
        C : array(d,d) or float
            Prior covariance
            
        """
    # Model Parameters
        self.beta = beta
        self.T = T
    
    # Discretisation Parameters
        self.n = n
        self.dt = T/n
        self.k = k # m observations at times k, 2k, 3k, ...
        
    # Observation operator (Attention: R^n+1 -> R^m )
        self.H = np.zeros((self.m,n+1))
        self.H[np.arange(0,self.m),np.arange(k,n+1,k)] = 1
            
        
    def __str__(self):
        out = f'beta{self.beta}T{self.T}n{self.n}k{self.k}'
        return out
    
    @property
    def name(self):
        name = 'conddiff'
        return name
    
    @property
    def m(self):
        m = int(self.n/self.k)
        return m
    
    @property
    def d(self):
        d = self.n 
        return d
    
    ###########################################
    def integrate_model(self, w):
    ###########################################
        """ Compute particle trajectory u given Brownian increments w
        
        Parameters
        ----------
        w : array(n)
            Array with n Brownian incremnts.

        Returns
        -------
        u: array(n+1)
           Particle path at n+1 time steps in [0,T]

        """
        u = np.zeros(self.n+1)

        for i in range(self.n):
            f = self.beta*u[i]*(1-u[i]**2)/(1+u[i]**2)
            u[i+1] = u[i] + self.dt*f + np.sqrt(self.dt)*w[i]
        
        return u
    
    ###########################################
    def compute_G(self, w):
    ###########################################
        """ Compute G = H_obs ° M 
        
        Parameters
        ----------
        w : array(n)
            Array with n Brownian incremnts.
        
        Returns
        -------
        G : array(m)
            Particle at obs locations
        """        
        u = self.integrate_model(w)
        G = self.H @ u
        return G
        
    
    ###########################################
    def compute_gradG(self, w, states=False):
    ###########################################
        """ Compute gradG = Jacobian G
        
        Computational trick is used: 
        Invert offdiagonal matrix instead of looping to make products in dudw

        Parameters
        ----------
        w : array(n)
            Array with n Brownian incremnts.

        Returns
        -------
        gradG : array(m,n)
            Returns gradient of generalised forward operator (H_obs * nabla_w(G))^T,
            i.e. model derivative w.r.t. parameter w (Brownian increment),
            multiplied by observation operator H_obs.

        """        
        u = self.integrate_model(w)
        N    = self.n+1
        tmp = u**2
        dfdu = self.beta*( (1 - 3*tmp)/(1 + tmp ) - 2*tmp*(1 - tmp)/(1 + tmp)**2 )
        off_diag  = -1-dfdu[:-1]*self.dt
        tmp = np.eye(N)
        tmp[np.arange(1,N),np.arange(0,N-1)] = off_diag
        dudw = np.sqrt(self.dt) * np.linalg.inv(tmp)[:,1:]
        # Note that u in R^n+1 while w in R^n so that dudw is in R^(n+1 x n)
        # Thus G in R^(m x n)
        gradG = self.H @ dudw
        
        if states:
            return u, gradG
        else:
            return gradG
    

###########################################
#%% Initalise Model
###########################################

def init_conddiff(**kwargs):
    beta = kwargs.get('beta',10)
    T = kwargs.get('T',1)
    n = kwargs.get('n',100)
    k = kwargs.get('k', 1)
    model = CondDiff(beta,T,n,k)
    return model

if __name__ == "__main__":
    model = init_conddiff()