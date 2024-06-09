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
from plotting import *

###########################################
# Classes
###########################################
class Burgers(model.Model):
    
    name = 'burgers'
    
    def __init__(self, N=100, T=1, D=0.005, o='periodic', dt=0.001, ks=1, kt=1, obs_init=False):
    # Model Parameters
        self.N = N
        self.T = T
        self.D = D
        self.o = o
        
    # Discretisation Parameters
        self.h = 1/N
        self.dt = dt
        self.n = int(self.T/self.dt)
        self.ks = ks
        self.kt = kt        
        self.nb_obs_space = int(self.N/self.ks)
        self.nb_obs_time = int(self.n/self.kt)
        self.obs_init = obs_init #observe initial conditions or not
            
    # System matrices
        self.Hobs = self.generate_Hobs()
        self.Mdiff = self.generate_Mdiff()
        self.Madv = self.generate_Madv()
        self.Mcentered = self.generate_Mcentered()
         
    def __str__(self):
        out = f'N{self.N}T{self.T}D{self.D}o{self.o}dt{self.dt}'
        out += f'ks{self.ks}kt{self.kt}obsinit{int(self.obs_init)}'
        return out
    
    @property
    def name(self):
        if self.nb_obs_time == 1:
            name = 'burgers_static'
        else:
            name = 'burgers'
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
    
    def generate_Mcentered(self):
        M =  np.zeros((self.N, self.N))
        for i in range(self.N-1):
            M[i+1,i] = -1
            M[i,i+1] = +1
        if self.o == 'periodic':
            M[0,-1] = -1
            M[-1,0] = 1
        M = self.dt/(2*self.h) * M
        return M
    
    def generate_Madv_0(self):
        ''' non-periodic Adjoint/TLM has 0 Dirichlet BC '''
        M = - np.eye(self.N)
        for i in range(self.N-1):
            M[i+1,i] = 1
        M = self.dt/self.h * M
        return M
    
    def generate_Mdiff_0(self):
        ''' non-periodic Adjoint/ TLM has 0 Dirichlet BC '''
        M = -2 * np.eye(self.N)
        for i in range(self.N-1):
            M[i,i+1] = 1
            M[i+1,i] = 1
        M = self.dt*self.D/self.h**2 * M
        return M
        
    ###########################################
    def integrate_model(self, u0, tf):
    ###########################################
        """ Integrates model starting from state u0 until time tf
    
        Parameters
        ----------
        u0 : array(u)
            Initial state
        tf : float
            Final time. 
        
        Returns
        -------
        states : array(nb_t, N)
            Row i = state u at time step t_i
            Includes initial state and state at final time

        """
        times = np.arange(0, tf+self.dt, self.dt)
        states = np.empty((len(times), self.N))
        states[0,:]=u0.copy()
        u = u0.copy()  
        for i in range(1,len(times)):
              u = u + self.Mdiff @ u \
                  - ((u>0) * np.abs(u)) * (self.Madv @ u) \
                  - ((u<0) * np.abs(u)) * (self.Madv.T @ u)
              states[i,:] = u.copy()
             
        return states
    
    ###########################################
    def integrate_TLM(self, du0, tf, states):
    ###########################################
        """ Integrates TLM starting from dx until time tf
                
        Used to compute gradG @ w, while integrate_b used for adjoint gradG.T @ w
        
        Attention: Not implementing discretisation of continuous TLM, but 
        rather directly linearise the discrete forward integration model!
        
        Parameters
        ----------
        du0 : array(N)
            Initial state in TLM. Corresponds to input perturbation direction
        tf : float
            Final time. 
        states : array(tf/dt, N)
            From forward integration. Becomes advection beta in TLM and Adjoint
        
        Returns
        -------
        states_TL : array(tf/dt, N)
            Row i = state of TLM u at time step t_i
            Includes initial state dx and state at final time
    
        """
        times = np.arange(0, tf+self.dt, self.dt)
        assert len(states) == len(times)-1
        states_TLM = np.empty((len(times), self.N))
        u_TL = du0.copy()
        states_TLM[0,:]=u_TL.copy()  
        
        for i in range(1,len(times)):
            u = states[i-1].copy()
            u_TL = u_TL + self.Mdiff @ u_TL \
                - (np.diag(((u>0) * np.abs(u))) @ self.Madv) @ u_TL \
                - (np.diag((u<0) * np.abs(u)) @ self.Madv.T) @ u_TL \
                + np.diag((u>0) * (-self.Madv @ u)  + (u<0) * (self.Madv.T @ u)) @ u_TL
                # Last line is reaction term in TLM
            states_TLM[i,:] = u_TL.copy()
            
        return states_TLM
    
    ###########################################
    def integrate_b(self, uf, tf, states):
    ###########################################
        """ Integrates Adjoint backward from dy at tf 
        
        Attention: Not implementing discretisation of Adjoint model, but 
        rather directly take adjoint of discrete TLM!
    
        Somehow check_Adj not working, I think integrate_b is correct tho
        
        Parameters
        ----------
        uf : array(N)
            Final state to start backward integration.
            Corresponds to misfit/nested grad/output perturbation direction, 
            or forcing in linear adjoint model
        tf : float
            Final time.
        states : array(tf/dt, N)
            From forward integration. Becomes advection beta in TLM and Adjoint
        
        Returns
        -------
        state : array(tf/dt, N)
            Row i is state of adjoint at time step tf-i
            Inculdes final state dy and state at t0

        """
        times = np.arange(0,tf+self.dt, self.dt)
        assert len(states) == len(times)-1 
        states_b = np.empty((len(times),self.N))
        u_b = uf.copy()
        states_b[-1] = u_b.copy()
                
        for i in range(1,len(times)):
            u = states[-i].copy()
            u_b = u_b + self.Mdiff @ u_b\
                - (np.diag((u>0) * np.abs(u)) @ self.Madv).T @ u_b\
                - (np.diag((u<0) * np.abs(u)) @ self.Madv.T).T @ u_b\
                + np.diag((u>0) * (-self.Madv @ u)  + (u<0) * (self.Madv.T @ u)) @ u_b
                # Last line is reaction term in TLM
            states_b[-(i+1)] = u_b.copy()
            
        return states_b
    
    
    ###########################################
    def compute_G(self, u):
    ###########################################
        """ Compute G = H_obs ° M
        
        Parameters
        ----------
        u: array(n) or (T/dt, N)
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
            states = u.copy()
            
        for i in np.arange(self.nb_obs_time):
            G[i] = self.Hobs @ states[(i+1-self.obs_init)*self.kt]
            
        return G.flatten()
    
    ###########################################
    def compute_gradG_w(self, w, states):
    ###########################################
        """ Compute gradG(u0) @ w with TLM
        
        Parameters
        ----------
        w : array(N)
            Direction of perturbation in initial conditions
        states : array(T/dt+1, N)
            From forward integration. Becomes advection beta in TLM.
            Contains both init and final state, though only init used

        Returns
        -------
        gradG : array(m) 
            Computes gradG @ w
            
        """
        assert len(states) == int(self.T/self.dt) + 1
        w = w.reshape(self.N)
        gradGw = np.empty((self.nb_obs_time, self.nb_obs_space)) 
        states_TL = self.integrate_TLM(w, self.T, states[:-1])
        
        for i in range(self.nb_obs_time):
            gradGw[i] = self.Hobs @ states_TL[(i+(1-self.obs_init))*self.kt]
            
        return gradGw.flatten()
    
    ###########################################
    def compute_gradGT_w(self, w, states):
    ###########################################
        """ Integrates Adjoint backward starting with 0 at tf and with forcing/misfit dy
    
        Parameters
        ----------
        w : array(nb_obs_time, nb_obs_space)
            Forcing in linear adjoint model. Corresponds to misfit/nested grad/output perturbation direction
        states : array(T/dt+1, N)
            From forward integration. Becomes advection beta in TLM
            Contains both init and final state, tho only finla used
        
        Returns
        -------
        gradGTw : array(d)
            Computes gradG^T @ w

        """
        assert len(states) == int(self.T/self.dt)+1
        w = w.reshape((self.nb_obs_time, self.nb_obs_space))
        gradGTw = self.Hobs.T @ w[-1]
        if not self.obs_init:
            u = states[-self.kt:]
            idx = 1
        else:
            u = states[-2*self.kt:-self.kt]
            idx = 2
        # since using i:-1 exludes last index of slice
        
        for i in range(idx, self.nb_obs_time):
            gradGTw = self.integrate_b(gradGTw, self.kt*self.dt, u)[0]
            gradGTw = gradGTw + self.Hobs.T @ w[-(i+(1-self.obs_init))]
            u = states[-(i+1)*self.kt: -(i*self.kt)]
        
        if not self.obs_init:
            gradGTw = self.integrate_b(gradGTw, self.kt*self.dt, u)[0]
        else:
            gradGTw = self.integrate_b(gradGTw, self.kt*self.dt, u)[0]
            gradGTw = gradGTw + self.Hobs.T @ w[0]
            
        return gradGTw
    
    ###########################################
    def compute_gradG(self,u0, method='TL'):
    ###########################################
        """ Compute full gradG(u0) using Adjoint or TLM
        
        Parameters
        ----------
        u0 : array(N) or (T/dt, N)
            Initial state or integrated states at all times

        Returns
        -------
        gradG : array(m,N)
            Computes gradG(u0)
            
        """
        if len(np.shape(u0)) == 1:
            states = self.integrate_model(u0, self.T)
        else:
            states = u0.copy()
        
        if method == 'TL': # construct full gradient using TLM
            gradG = np.empty((self.m,self.N))
            for i in range(self.N):
                    exact_direction = np.zeros(self.N)
                    exact_direction[i] = 1
                    gradG[:,i] = self.compute_gradG_w(exact_direction, states)
                    
        elif method == 'Adjoint':
            gradG = np.empty((self.nb_obs_time, self.nb_obs_space,self.N))
            for i in range(self.nb_obs_time):
                for j in range(self.nb_obs_space):
                    exact_direction = np.zeros((self.nb_obs_time,self.nb_obs_space))
                    exact_direction[i,j] = 1
                    gradG[i,j,:] = self.compute_gradGT_w(exact_direction, states)
            gradG = gradG.reshape((self.m,self.N))
        
        return gradG
    
    ###########################################
    def check_TLM(self, x, seed = None):
    ###########################################
        """ Compare finite difference (G(x)-G(x+eps*dx))/eps with TLM G[x](dx) """
        if seed is None:
            dx = np.random.randn(self.d)
        else:
            dx = seed
        dx = dx/np.linalg.norm(dx)
        u = self.integrate_model(x, self.T)
        TLM = self.integrate_TLM(dx, self.T, u[:-1])
        error = np.empty(10)
        for i in range(10):
            alpha = 10.**(-(i+1))
            u0_pert = x + alpha*dx
            u_pert = self.integrate_model(u0_pert, self.T)
            finite_diff = (u_pert-u)/alpha
            error[i] = np.linalg.norm(finite_diff -  TLM)/np.linalg.norm(finite_diff)
            print('alpha = ',alpha, 'error = ',error[i])
        return error
            
    ###########################################
    def check_Adjoint(self, x, nb = 10, seed_dx = None, seed_dy = None):
    ###########################################
        """ Compare <gradG(x) @ dx, dy> with <dx, gradG(x).T @ dy>     """
        u = self.integrate_model(x, self.T)
        if seed_dx is not None:
            assert len(seed_dx) == len(seed_dy)
            nb = len(seed_dx)
        
        error = np.empty(nb)
        for i in range(nb):
            if seed_dx is None:
                dx = np.random.randn(self.d)
            else:
                dx = seed_dx[i]
            if seed_dy is None:
                dy = np.random.randn(self.nb_obs_time,self.nb_obs_space)
            else:
                dy = seed_dy[i]
            dx = dx/np.linalg.norm(dx)
            dy = dy/np.linalg.norm(dy)
            
            G_TLM = self.compute_gradG_w(dx, u)
            G_Adj = self.compute_gradGT_w(dy, u)
            res1 = G_TLM @ dy.flatten()
            res2 = G_Adj @ dx
            error[i] = np.abs(res1-res2)/(np.abs(res1) + np.abs(res2))
            print('error : ', error[i])
        return error
        
            
    ###########################################
    def check_gradG(self,x=None, method='TL'):
    ###########################################
        """ Adapted from shallow water check_grad 
        
        Componentwise for vector valued fct instead of scalar cost_fct.
        """  
        if x is None:
            x = np.random.randn(self.d)
        G = self.compute_G(x)
        grad = self.compute_gradG(x, method=method)
        errors = np.empty((10, self.m))
        for i in range(10):
            alpha = 10.**(-(i+1))
            G_pert = np.zeros(self.m)
            error = np.zeros(self.m)
            for j in range(self.m):
                u0_pert = x + alpha*grad[j]/np.linalg.norm(grad[j])
                G_pert[j] = self.compute_G(u0_pert)[j]
                error[j] = np.abs(G[j] - G_pert[j])/ (alpha* np.linalg.norm(grad[j]))
            print('alpha = ',alpha, 'error = ',np.mean(error))
            errors[i] = error.copy()
            # should go to 1 
        return errors

###########################################
#%% Initalise Model
###########################################

def init_burgers(**kwargs):
    obs_init = kwargs.get('obs_init', True)
    N = kwargs.get('N', 100)
    T = kwargs.get('T', 1)
    D = kwargs.get('D', 0.001)
    o = kwargs.get('o', 'periodic')
    dt = kwargs.get('dt', 0.001)
    ks = kwargs.get('ks', 10) 
    kt = kwargs.get('kt', 100)
    model = Burgers(obs_init=obs_init, N=N, T=T, D=D, o=o, dt=dt, ks=ks, kt = kt)
    return model

if __name__ == "__main__":
    model = init_burgers()