#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Class for Gaussian Distribution definition """

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz #for matern
from scipy.special import gamma, kv #for matern

###########################################
#%% Class
###########################################

class Gaussian:
    
    def __init__(self, mu = 0, Sig=1, d=1, d_red=0, **kwargs):
        
        self.d = d
        self.mu_name, self.mu = self.setup_mu(mu)
        if isinstance(Sig, (int,float)): 
            self.Sig_name = f'{Sig}'
            Sig, Sig12, Sigm1, Sigm12, d_red, poincare, cramerrao = self.setup_scalar_cov(Sig)
        elif isinstance(Sig,str):
            self.Sig_name = 'Matern'
            Sig = self.setup_matern_Sig(**kwargs)
            Sig, Sig12, Sigm1, Sigm12, d_red, poincare, cramerrao = self.setup_matrix_cov(Sig, d_red=d_red)
        elif isinstance(Sig, np.ndarray):
            self.Sig_name = 'Matrix'
            self.d = np.shape(Sig)[0]
            Sig, Sig12, Sigm1, Sigm12, d_red, poincare, cramerrao = self.setup_matrix_cov(Sig, d_red=d_red)
        self.d_red = d_red
        self.Sig = Sig
        self.Sig12 = Sig12
        self.Sigm1 = Sigm1
        self.Sigm12 = Sigm12
        self.poincare = poincare
        self.cramerrao = cramerrao
        
        self._rv = multivariate_normal(mean=self.mu, cov=self.Sig)
        self.pdf = self._rv.pdf
        self.rvs = self._rv.rvs
        
    def __str__(self):
        return self.Sig_name
    
    def setup_mu(self, mu):
        if isinstance(mu, (int,float)):
            name = mu
            mu = mu * np.ones(self.d)
        elif isinstance(mu,np.ndarray):
            name = 'Array'
            mu = mu
        elif mu == 'gauss':
            name = mu
            def gauss(x):
                a = 100
                mean = self.d/2
                sig = 0.1*self.d
                res = a/np.sqrt(2*np.pi*sig**2)*np.exp(-(mean-x)**2/(2*sig**2))
                return res
            mu = np.array([gauss(i) for i in range(self.d)])
        elif mu == 'sin':
            name = mu
            def sin(x):
                return np.sin(2*np.pi*x/self.d)
            mu = np.array([sin(i) for i in range(self.d)])
        return name, mu
    
    def setup_scalar_cov(self, Sig):
        d_red = self.d
        Sig = Sig
        Sig12 = np.sqrt(Sig)
        Sigm1 = 1/Sig
        Sigm12 = 1/np.sqrt(Sig)
        poincare = Sig
        cramerrao = Sig
        return Sig, Sig12, Sigm1, Sigm12, d_red, poincare, cramerrao
        
    def setup_matrix_cov(self,Sig, d_red=0):
        ev,ew,_ = np.linalg.svd(Sig)
        if d_red==0:
            d_red = np.argmax([ew/ew[0] < 10e-6])
            if d_red ==0:
                d_red = self.d
        Sig12 = ev[:,:d_red] @  np.diag(np.sqrt(ew[:d_red]))
        Sigm12 =  ev[:,:d_red] @ np.diag(1/np.sqrt(ew)[:d_red])
        Sigm1 = ev[:,:d_red] @ np.diag(1/ew[:d_red]) @ ev[:,:d_red].T
        poincare = np.max(ew)
        cramerrao = np.min(ew)
        return Sig, Sig12, Sigm1, Sigm12, d_red, poincare, cramerrao
    
    def setup_matern_Sig(self,**kwargs):
        N = self.d
        l = kwargs.get('l', 0.1)
        v = kwargs.get('v', 2.5)
        sig = kwargs.get('sig', 1)
        mode = kwargs.get('mode', 'periodic')
        dist = np.linspace(0,1,N+1)[:-1]
        dist = toeplitz(dist)
        if mode == 'periodic':
            Sig = matern_kernel(dist, l = l, v = v, sig=sig)
            for i in range(1,10):
                Sig += matern_kernel(dist + i,l = l, v = v)
                Sig += matern_kernel(dist - i,l = l, v = v)
        else:                
            Sig = matern_kernel(dist,l = l*N, v = v)
        return Sig
    
###############################################
#%% Functions
##############################################

def matern_kernel(dist, l = 4, v = 2.5, sig = 1):
    #v=2.5 twice diffbar
    #v=1.5 once diffbar
    dist = np.abs(dist)
    dist[dist==0]=1e-8
    part1 = sig**2 * 2 ** (1 - v) / gamma(v)
    part2 = (np.sqrt(2 * v) * dist / l) ** v
    part3 = kv(v, np.sqrt(2 * v) * dist / l)
    return part1 * part2 * part3