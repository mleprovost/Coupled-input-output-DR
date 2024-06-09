#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
import goals_setup as gs

import numpy as np
from dataclasses import dataclass

###########################################
#%% Classes
###########################################

@dataclass
class Sensor:
    name: str
    V: np.ndarray #for subspace
    maxdiag: np.ndarray = None
    EIM: np.ndarray = None
    
class SensorPlacement:
    
    def __init__(self, ew, V, D=None, **kwargs):
        self.nb_time = kwargs.get('nb_time', None)
        self.nb_space = kwargs.get('nb_space',None)
        self.s = np.shape(V)[1]
        if self.nb_space is not None:
            V = np.reshape(V,(self.nb_obs_time,self.nb_obs_space,self.s))
            D = np.reshape(D,(self.nb_obs_time,self.nb_obs_space)) 
        self.m, self.s = np.shape(V)
        self.nb = self.s
        self.ew = ew # array(s)
        self.V = V # array(m,s)
        self.D = D # array(m) Diagonal entries of H_Y(U_r)
        
    def maxvals_diag(self):
        """ 
        Returns indices of largest entries of diag(H_Y(U_r))
        
        """
        sensors = np.argsort(self.D)[-self.nb:]
        return sensors
    
    def EIM_ordered(self):
        """
        Hierarchical basis given as ordered columns of V.
        For invertibility of B need columns of V to be linear indep
        
        Souce
        -----
        Y. Maday, N.C. Nguyen, A.T. Patera: A general, multipurpose
            interpolation procedure: the magic points

        """
        x = np.zeros(self.nb, int) #sensor locations x_i
        B = np.zeros((self.nb,self.nb))
        Q = np.zeros((self.m, self.nb)) #columns of Q contain q_j, Q_ij = q_j(x_i)
        
        x[0] = np.argmax(np.absolute(self.V[:,0]))
        Q[:,0] = self.V[:,0]/self.V[x[0],0]
        B[0,0] = Q[x[0],0]
        #Although entries overlap, we need both matrices B and Q since B is i x i while Q has i full columns
        
        for i in range(1,self.nb):
            beta = np.linalg.solve(B[:i,:i], self.V[x[:i],i])     
            x[i] = np.argmax(np.absolute(self.V[:,i] - Q[:,:i] @ beta))
            Q[:,i] = (self.V[:,i] - Q[:,:i] @ beta)/(self.V[x[i],i] - Q[x[i],:i] @ beta)
            #B[:i,i] = Q[x[:i],i] always 0 since B lower triangular with 1 diag
            assert np.allclose(Q[x[:i],i], np.zeros(i))
            B[i,:i] = Q[x[i],:i]
            B[i,i] = Q[x[i],i]
            assert np.allclose(B[:i+1,:i+1], np.tril(B[:i+1,:i+1]))
            assert np.allclose(np.diag(B[:i+1,:i+1]), np.ones((i+1)))
        if __debug__:
            W = np.zeros((self.m,self.nb))
            W[x,np.arange(self.nb)] = 1
            assert np.all(np.isclose(self.V @ np.linalg.inv(W.T@self.V), Q@np.linalg.inv(B)))
        return x
    
    def EIM_unordered(self, norm = None):
        """
        Magic points. 
        Basis chosen from ew weighted columns of V.
    
        Parameters
        ----------
        V : array(m,s)
            Full output projector. ATTENTION: needs to be orthonormal!
        
        Souce
        -----
        Y. Maday, N.C. Nguyen, A.T. Patera: A general, multipurpose
            interpolation procedure: the magic points
        """
        u = np.zeros(self.nb, int) #index of basis functions u_i
        x = np.zeros(self.nb, int) #sensor locations x_i
        B = np.zeros((self.nb, self.nb))
        Q = np.zeros((self.m, self.nb)) #columns of Q contain q_j, Q_ij = q_j(x_i)
        E = (np.diag(self.ew) @ self.V.T).T
        
        if norm is None or norm == 'L1':
            u[0] = np.argmax(np.sum(np.absolute(E), axis=0))
            x[0] = np.argmax(np.absolute(E[:,u[0]]))
        elif norm == 'Linfty':
            u[0] = np.argmax(np.sum(np.absolute(E), axis=0))
            x[0] = np.argmax(np.absolute(E[:,u[0]]))
        Q[:,0] = E[:,u[0]]/E[x[0],u[0]]
        B[0,0] = Q[x[0],0]
        
        for i in range(1,self.nb):
            Bm1 = np.linalg.inv(B[:i,:i]) #easier to assemble A with Bm1 instead of beta
            A =  np.array([E[:,j] - Q[:,:i] @ Bm1 @ E[x[:i],j]  for j in range(self.s)]).T
            
            if norm == 'L1':
                u[i] = np.argmax(np.sum(np.absolute(A), axis=0))
                x[i] = np.argmax(np.absolute( E[:,u[i]] - Q[:,:i] @ Bm1 @ E[x[:i],u[i]] ))
            elif norm == 'Linfty':
                norms = np.max(np.absolute(A), axis=0)
                norms[u[:i]] = 0
                u[i] = np.argmax(norms)
                x[i] = np.argmax(np.absolute( E[:,u[i]] - Q[:,:i] @ Bm1 @ E[x[:i],u[i]] ))
            
            Q[:,i] = (E[:,u[i]] - Q[:,:i] @ Bm1 @ E[x[:i],u[i]])/(E[x[i],u[i]] - Q[x[i],:i] @ Bm1 @ E[x[:i],u[i]])
            B[:i,i] = Q[x[:i],i] #always 0 since B lower triangular
            B[i,:i] = Q[x[i],:i]
            B[i,i] = Q[x[i],i] #always 1 since B has 1 on diag
            assert np.allclose(B[:i+1,:i+1], np.tril(B[:i+1,:i+1]))
            assert np.allclose(np.diag(B[:i+1,:i+1]), np.ones((i+1)))
        if __debug__:
            W = np.zeros((self.m,self.nb))
            W[x,np.arange(self.nb)] = 1
            assert np.all(np.isclose(self.V @ np.linalg.inv(W.T@self.V), Q@np.linalg.inv(B)))
        return x

##############
#%% Functions
##############

###############################################################################
def compute_sensor(res, nb_sensors, methods, name = None, dr=None, precond=True, load=True, **kwargs):
###############################################################################  
    """ Compute dict of sensors indexed by their goal names

    Parameters
    ----------
    res : list/tuple
        results [HX,HY,iter_U,iter_V] or [V,ew] from PCA
    name : str
        goal name or PCA (since PCA sensors independent of goal)
    methods : list
        which sensor placement methods are used
    **kwargs : 
        parameters required for specific sensor placment methods

    Returns
    -------
    sensors : Sensor
        Sensor object containing placments with different methods

    """ 
    if isinstance(res,gs.Goal): 
        goal = res
        r = np.shape(goal.U)[1]
        HX, HY, iter_U,iter_V = dr.compute_coupled(r=r, s=nb_sensors, U = goal.U, \
                                max_iter=1, precond=precond, load = load, filename = f'{dr.results_folder}/M{dr.M}_goal_{goal.name}_r{r}s{nb_sensors}')
        V =  iter_V[-1]
        _,ew,_ = np.linalg.svd(HY)
        s = np.shape(V)[1]
        ew = ew[:s]
        D = np.diag(HY)
        name = goal.name
    elif len(res) == 4:
        HX, HY, iter_U,iter_V = res
        V =  iter_V[-1]
        _,ew,_ = np.linalg.svd(HY)
        s = np.shape(V)[1]
        ew = ew[:s]
        D = np.diag(HY)
    elif len(res) == 2: #PCA
        V, ew = res #dr.compute_PCA_out()
        V = V[:,:nb_sensors]
        ew = ew[:nb_sensors]
        D = None
    
    estimator = SensorPlacement(ew=ew, V=V, D=D, **kwargs)
    
    sensor = Sensor(name = name, V=V) # Initiate first since attributes optional
    for method in methods:        
        if method == 'maxdiag':
            sensor.maxdiag = estimator.maxvals_diag()
        elif method == 'EIM_ordered':
            sensor.EIM = estimator.EIM_ordered()
        elif method == 'EIM_unordered':
            sensor.EIM = estimator.EIM_unordered(norm=kwargs.get('norm',None))
                
    return sensor