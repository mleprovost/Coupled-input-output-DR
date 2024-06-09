#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
from models import model

import numpy as np
import scipy as sp
import utils 

###########################################
# Classes
###########################################
class AdvDiff(model.Model):
    
    def __init__(self, N = 100, T=1, D=0.005, beta=2, o='periodic', dt=0.001, ks=1, kt=1, obs_init = False):
    # Model Parameters 
        self.T = T
        self.D = D
        self.beta = beta
        self.o = o
        
    # Discretisation Parameters
        self.N = N
        self.h = 1/N
        self.dt = dt
        self.n = int(self.T/self.dt)
        self.ks = ks
        self.kt = kt      
        self.obs_init = obs_init
        self.nb_obs_space = int(self.N/self.ks)
        self.nb_obs_time = int(self.n/self.kt)

    # Check Peclet and CFL condition
        if not self.h * np.min(np.abs(self.beta)) / self.D >= 2: 
            raise Exception(f"Peclet condition (h*beta/D = {self.h*np.min(np.abs(beta))/D} >= 2) not satisfied")
        if not self.dt <= self.h/np.max(np.abs(self.beta)): 
            raise Exception(f"CFL condition (dt = {dt} =< h/beta = {self.h/np.max(np.abs(beta))}) not satisfied")
        
    # System matrices
        self.Hobs = self.generate_Hobs()
        self.M_diff = self.generate_Mdiff()
        self.M_adv = self.generate_Madv()
        if isinstance(self.beta, (int,float)):
            self.M = np.eye(self.d) + self.M_diff - (self.beta>0)*np.abs(self.beta)*self.M_adv - (self.beta<0)*np.abs(self.beta)*self.M_adv.T
        elif isinstance(self.beta, np.ndarray):
            self.M = np.eye(self.d) + self.M_diff - (self.beta>0)*np.diag(np.abs(self.beta)) @ self.M_adv - (self.beta<0)*np.diag(np.abs(self.beta)) @ self.M_adv.T
        self.M_kt = np.linalg.matrix_power(self.M,self.kt)

    def __str__(self):
        out = f'N{self.N}T{self.T}D{self.D}beta{self.beta}o{self.o}dt{self.dt}'
        out += f'ks{self.ks}kt{self.kt}obsinit{int(self.obs_init)}'
        return out
    
    @property
    def name(self):
        if self.nb_obs_time == 1:
            name = 'advdiff_static'
        else:
            name = 'advdiff'
        return name
    
    @property
    def m(self):
        m = self.nb_obs_space*self.nb_obs_time 
        return m
    
    @property
    def d(self):
        d = self.N
        return d
    
    def generate_Hobs(self):
        Hobs = np.zeros((self.nb_obs_space,self.N)) 
        Hobs[np.arange(self.nb_obs_space),np.arange(0,self.N,self.ks)] = 1
        return Hobs
        
    def generate_Mdiff(self):
        M = -2 * np.eye(self.N)
        for i in range(self.N-1):
            M[i,i+1] = 1
            M[i+1,i] = 1
        if self.o == 'periodic':
            M[-1,0] = 1
            M[0,-1] = 1
        M = self.dt*self.D/self.h**2 * M
        return M
    
    def generate_Madv(self):
        M = np.eye(self.N)
        for i in range(self.N-1):
            M[i+1,i] = -1
        if self.o == 'periodic':
            M[0,-1] += -1
        M = self.dt/self.h * M
        return M
        
    ###########################################
    def integrate_model(self, u0, tf):
    ###########################################
        """ Integrates model starting from state u0 until time tf
    
        Parameters
        ----------
        u0 : array(u)
            Initial state
        tf : int
            Final time 
        
        Returns
        -------
        states : array(nb_t, n)
            Row i = state u at time step t_i

        """
        times = np.arange(0, tf+self.dt, self.dt)
        states = np.zeros((len(times), self.N))
        states[0,:]=u0.copy()
        u = u0.copy()        
        for i in range(1,len(times)):
             u = self.M @ u
             states[i,:] = u.copy()
             
        return states
        
    ###########################################
    def compute_G(self, u):
    ###########################################
        """ Compute G = H_obs ° M
        
        Parameters
        ----------
        u : array(n) or (T/dt,n)
            Initial state or integrated states at all times

        Returns
        -------
        G : array(m) = array(nb_obs_time,nb_obs_space).flatten()
            G with nb_obs_space at obs times
        """        
        G = np.zeros((self.nb_obs_time,self.nb_obs_space))
        if len(np.shape(u)) == 1:
            states = self.integrate_model(u, self.T)
        else:
            states = u
           
        for i in range(self.nb_obs_time):
            G[i] = self.Hobs @ states[(i+1-self.obs_init)*self.kt]
          
        return G.flatten()
    
    ###########################################
    def compute_gradG_w(self, w, states=0):
    ###########################################
        """ Compute gradG @ w

        Parameters
        ----------
        w : array(N)
            Direction of perturbation in initial conditions
        states: 
            To get uniform form as burgers
            
        Returns
        -------
        gradGw : array(m) 
            Column j = partial derivatives w.r.t X_j
            Recall m = nb_obs_time * nb_obs_space 
            Rows going (t1,k),(t1,2k), ... (t1,nb_obs_space*k),(t2,k),...(T,nb_obs_space*k)
            
        """
        gradGw = np.empty((self.nb_obs_time, self.nb_obs_space))
        ProductOfGradientsW = w.reshape(self.N)
        
        if self.obs_init:
            gradGw[0] = self.Hobs @ ProductOfGradientsW

        for i in range(self.nb_obs_time-self.obs_init):
            ProductOfGradientsW = self.M_kt @ ProductOfGradientsW
            gradGw[i+self.obs_init,:] = self.Hobs @ ProductOfGradientsW
                        
        return gradGw.flatten()
    
    ###########################################
    def compute_gradGT_w(self, w, states=0):
    ###########################################
        """ Compute gradG.T @ w

        Parameters
        ----------
        w : array(m)
            Direction of perturbation in observations
            
        Returns
        -------
        gradG : array(N) 
            gradG.T @ w
            
        """
        w = np.reshape(w,(self.nb_obs_time, self.nb_obs_space))
        gradG = self.Hobs.T @ w[-1]
            
        for i in range(2,self.nb_obs_time + 1):
            gradG = self.M_kt.T @ gradG
            gradG += self.Hobs.T @ w[-i]
            
        if not self.obs_init:
            gradG = self.M_kt.T @ gradG
            
        return gradG
    
    ###########################################
    def compute_gradG(self, u0=0):
    ###########################################
        """ Compute gradG = Jacobian G

        Returns
        -------
        gradG : array(m,d) 
            Column j = partial derivatives w.r.t X_j
            Recall m = nb_obs_time * nb_obs_space 
            Rows going (t1,k),(t1,2k), ... (t1,nb_obs_space*k),(t2,k),...(T,nb_obs_space*k)
            
        """
        gradG = np.zeros((self.nb_obs_time, self.nb_obs_space,self.N))
        
        ProductOfGradients = np.eye(self.N)
        times = range(0,self.nb_obs_time)
        if self.obs_init:
            gradG[0,:,:] = self.Hobs 
            times = range(1,self.nb_obs_time)
            
        for i in times:
            ProductOfGradients = self.M_kt @ ProductOfGradients
            gradG[i,:,:] = self.Hobs @ ProductOfGradients
            
        return gradG.reshape((self.m, self.N))
    
    ###########################################
    def check_Adj(self, nb = 10, seed_x = None, seed_y = None):
    ###########################################
        """ Compare <gradG @ x, w > with <x, gradG.T @ w>     """
        if seed_x is not None:
            assert len(seed_x) == len(seed_y)
            nb = len(seed_x)
        
        error = np.empty(nb)
        for i in range(nb):
            if seed_x is None:
                dx = self.Cm12 @ np.random.randn(self.d_pr)
            else:
                dx = seed_x[i]
            if seed_y is None:
                dy = np.random.randn(self.nb_obs_time,self.nb_obs_space)
            else:
                dy = seed_y[i]
            dx = dx/np.linalg.norm(dx)
            dy = dy/np.linalg.norm(dy)
            
            G_TLM = self.compute_gradG_w(dx)
            G_Adj = self.compute_gradGT_w(dy)
            res1 = G_TLM @ dy.flatten()
            res2 = G_Adj @ dx
            error[i] = np.abs(res1-res2)/(np.abs(res1) + np.abs(res2))
            print('error : ', error[i])
        return error
    
    
###########################################
#%% Initalise Model
###########################################

def init_advdiff(**kwargs):
    obs_init = kwargs.get('obs_init', True)
    N = kwargs.get('N', 100)
    T = kwargs.get('T', 1)
    D = kwargs.get('D', 0.001)
    beta = kwargs.get('beta', 2)
    o = kwargs.get('o', 'periodic')
    dt = kwargs.get('dt', 0.001)
    ks = kwargs.get('ks', 10) #20
    kt = kwargs.get('kt', 100) #200
    model = AdvDiff(obs_init=obs_init, N=N, T=T, D=D, beta=beta, o=o,dt=dt, ks=ks, kt = kt)
    return model

if __name__ == "__main__":
    model = init_advdiff()