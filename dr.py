#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__docformat__ = 'reStructuredText'

###########################################
# Imports
###########################################
import pickle # to save results
import os # to create folders
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds

import models.conddiff as cd
import models.burgers as bg
import models.advdiff as ad
import bayes.bip as bip
import utils

###########################################
# Classes
###########################################
class DR:
    def __init__(self, bayes, M=10, full_grad = True,  load = True, save = True, cache_tag = ''):
        """
        Performs computation of Hs on given model.

        Parameters
        ----------
        model : needs attributes compute_G and compute_gradG or compute_G_gradG
        M : int
            Number of prior samples for Hs computation
        
        """
        self.bayes = bayes
        self.full_grad = full_grad
        self.d = bayes.d
        self.d_red = self.bayes.d_red
        self.m = bayes.m
        self.M = M
        
        # Setup results and plots folder
        self._results_folder = f'results{cache_tag}/{self.model.name}/{self.bayes}'
        self._plots_folder = f'plots{cache_tag}/{self.model.name}/{self.bayes}'
        
        # Compute prior_samples, states
        self.prior_std, self.prior_samples = self.bayes.generate_prior_samples(M=self.M, load = load, save=save)
        if self.model.name != 'blockdiff':
            self.states = self.bayes.generate_state_samples(self.prior_samples, load=load, save=save)
        if self.model.name in ['conddiff', 'blockdiff']:
            samples = self.prior_samples
        else:
            samples = self.states
            
        # Compute G and gradG samples
        if self.full_grad:
            if self.bayes.G_gradG:
                self.G, self.gradG = self.bayes.generate_G_gradG_samples(samples,M=self.M, load=load, save=save)
            else: 
                self.G = self.bayes.generate_G_samples(samples, M=self.M, load = load, save=save)
                self.G_lin = self.bayes.generate_G_lin()
                if self.model.name[:7] == 'advdiff':
                    self.gradG = [self.G_lin]*self.M
                else:
                    self.gradG = self.bayes.generate_gradG_samples(samples, M=self.M, load = load, save=save)
        else:
            self.G = self.bayes.generate_G_samples(samples, M=self.M, load = load, save=save)            
                
        # Setup matvec functions
        if self.model.name != 'conddiff':
            def compute_gradG_w(w,states=0):
                return np.dot(self.model.compute_gradG_w(w,states=states),self.bayes.like.Sigm12).T
            def compute_gradGT_w(w,states=0):
                return self.model.compute_gradGT_w(np.dot(w,self.bayes.like.Sigm12).T,states=states)
            self.compute_gradG_w = compute_gradG_w
            self.compute_gradGT_w = compute_gradGT_w
            
        # Setup prior sample cov
        self._C_sample = None
        
        
    @property
    def model(self):
        return self.bayes.model
    
    @property
    def C_sample(self):
        if self._C_sample is None:
            self._C_sample = np.cov(self.prior_samples.T)
        return self._C_sample
    
    @property
    def results_folder(self):
        # Use property to only create folder if really needed
        os.makedirs(self._results_folder, exist_ok=True)
        return self._results_folder
    
    @property
    def plots_folder(self):
        # Use property to only create folder if really needed
        os.makedirs(self._plots_folder, exist_ok=True)
        return self._plots_folder
    
    
    ##########################################
    def compute_PCA_in(self, sample=False):
    #########################################
        """ Compute PCA of input
        
        Parameters
        ----------
        samples : bool
            If true use sample cov, else use bg cov
        
        """
        if sample: 
            ev, ew , _ = np.linalg.svd(self.C_sample)
        else:
            if isinstance(self.bayes.prior.Sig,(int,float)):
                ev, ew , _ = np.linalg.svd(np.diag([self.bayes.prior.Sig]*self.d))
            elif isinstance(self.bayes.prior.Sig, np.ndarray):
                ev, ew , _ = np.linalg.svd(self.bayes.prior.Sig)
        return ev, ew
        
    ##########################################
    def compute_PCA_out(self):
    #########################################
        """ Compute PCA of output """
        ev, ew , _ = np.linalg.svd(np.cov(self.G.T))
        return ev, ew
    
    
    

    #########################################################
    def compute_HX(self, precond = False):
    #########################################################
        """ Computes HX or whitened HX_bar
        
        HX = int( gradG.T @ Rm1 @ Rm1 @ gradG, dprior)
        HX_bar = C12.T @ int( gradG.T @ Rm1 @ gradG, dprior) @ C12
        
        Here not computing reduced form since toy problems assume m,d small and 
        want to use M very large for good MC estimate. 
        
        Dimension use d_bg instead of d, in case prior cov C12 was truncated
        Usefull to check correct transpose and preconditioning of matrices
        
        Note that we set G(x) = R12 @ G(x) so that self.R is always 1
        
        Parameters
        ----------
        precond : bool
            If true compute HX_bar, else compute HX.
            HX_bar needs generalised EV (HX_bar, Sigm1), HX needs normal EV.
            
        Source
        ------
        R. Baptista, O. Zahm: Gradient-based data and parameter dimension
            reduction for Bayesian models: an information theoretic perspective
        """
        print('Computing HX')          
        if precond: 
            HX = np.zeros((self.d_red,self.d_red))
        else:
            HX = np.zeros((self.d, self.d))
            
        for i in range(self.M):
            if precond:
                tmp = np.dot(self.gradG[i],self.bayes.prior.Sig12)
            else: 
                tmp = self.gradG[i]
            HX = HX + 1/self.M * tmp.T @ tmp

        return HX
    
    #########################################################
    def compute_HY(self, precond = False):
    #########################################################
        """ Computes HY or whitened HY_bar
        
        HY = int( gradG @ gradG.T, dprior) 
        HY_bar = Rm12 @ int( gradG @ C @ gradG.T, dprior) @ Rm12
        
        Parameters
        ----------
        precond : bool
            If true compute HY_bar, else compute HY.
            HY_bar needs generalised EV (HY_bar, R), HY needs normal EV.
            
        Source
        ------
        R. Baptista, O. Zahm: Gradient-based data and parameter dimension
            reduction for Bayesian models: an information theoretic perspective
        """
        print('Computing HY')                
        HY = np.zeros((self.m,self.m))
        for i in range(self.M):
            if precond:
                tmp = np.dot(self.gradG[i], self.bayes.prior.Sig12)
            else: 
                tmp = self.gradG[i]
            HY = HY + 1/self.M * tmp @ tmp.T
            
        return HY
    
    #########################################################
    def compute_HX_V(self, V, precond = False, full_grad=True):
    #########################################################
        """ Computes HX(V) or precond version
        
        HX(V) = int( gradG.T @ V @ V.T @ gradG, dprior)
        HX(V)_bar = C12.T @ int( gradG.T @ Rm12 @ V @ V.T @ Rm12 @ gradG, dprior) @ C12
        
        Note that we set G(x) = R12 @ G(x) so that self.R is always 1
        
        """
        print('Computing HX(V)')   
        s = np.shape(V)[1]
        if full_grad:
            if not precond:
                HX = np.zeros((self.d,self.d))
            else:
                HX = np.zeros((self.d_red,self.d_red))  
            for i in range(self.M):
                if not precond:
                    tmp = self.gradG[i].T @ V
                else:
                    tmp = np.dot(self.bayes.prior.Sig12.T, self.gradG[i].T) @ V
                HX += (1/self.M) * tmp @ tmp.T
        else:
            if not precond:
                d = self.d
                def matvec(v):
                    res = np.zeros(d)
                    for i in range(self.M):
                        res += 1/self.M * self.compute_gradGT_w( V @ (V.T @ \
                            self.compute_gradG_w(v, states=self.states(i))), states=self.states(i))
                    return res
            else:
                d = self.d_red
                def matvec(v):
                    res = np.zeros(d)
                    for i in range(self.M):
                        res += 1/self.M * np.dot(self.bayes.prior.Sig12.T, self.compute_gradGT_w(V @ (V.T @ \
                            self.compute_gradG_w( np.dot(self.bayes.prior.Sig12,v), states=self.states(i))), states=self.states(i)))
                    return res
            HX = LinearOperator(shape=(d,d), matvec = matvec, rmatvec = matvec)            
        return HX

    #########################################################
    def compute_HY_U(self, U, precond = False, full_grad=True):
    #########################################################
        """ Computes HY(U) or preconditioned version
        
        HY(U) = int( gradG @ U @ U.T @ gradG.T, dprior)
        HY(U)_bar = Rm12 @ int( gradG @ C12 @ U @ U.T @ C12.T @ gradG.T, dprior) @ Rm12
        
        """
        print('Computing HY(U)')
        r = np.shape(U)[1]
        if precond:
            U = np.dot(self.bayes.prior.Sig12,U)
        if full_grad:
            HY = np.zeros((self.m,self.m))
            for i in range(self.M):
                tmp = self.gradG[i] @ U
                HY += (1/self.M) * tmp @ tmp.T
        else:
            def matvec(v):
                res = np.zeros(self.m)
                for i in range(self.M):
                    res += 1/self.M * self.compute_gradG_w(U@ (U.T @ \
                        self.compute_gradGT_w(v, states = self.states(i))), states=self.states(i))
                return res
            HY = LinearOperator(shape=(self.m,self.m), matvec = matvec, rmatvec = matvec) 
            
        return HY
    
    #########################################################
    def compute_coupled(self, r, s, U=None, V=None, max_iter=10, precond = False, full_grad=True, load=True, filename = ''):
    #########################################################
        """ Alternating Eigendecomposition
        
        Attention: If provided, shape of U has to be (d_bg, r) in case of precond=True
        since prior covariance might be truncated
        
        If no U, V provided, start alternEIG by default with V 
        
        """
        if self.full_grad == False: full_grad = False
        # Make sure file under standard filename not overwritten if specific U/V supplied 
        save = True
        if (U is not None or V is not None) and filename == '':
            save = False
        if filename == '':
            filename = f'{self.results_folder}/M{self.M}r{r}s{s}precond{precond}maxiter{max_iter}fullgrad{int(full_grad)}_alternSVD'
            
        if load:
            if os.path.isfile(filename):
                with open(f'{filename}', 'rb') as f: 
                    HX, HY, iter_U, iter_V = pickle.load(f)
                print(f'Loaded alternating SVD results from {filename}')
            else:
                print(f'No coupled results found in {filename} \nProceed to computation')
                load = False
                
        if not load:
            iter_U = []
            iter_V = [] 
            HX = None
            HY = None
            
            # Initialise U,V for alternSVD #############################
            if U is not None:
                init_U = True
                V = np.zeros((self.m,s))
                r = np.shape(U)[1]
            elif V is not None:
                init_U = False
                if not precond: 
                    U = np.zeros((self.d,r))
                else:
                    U = np.zeros((self.d_red,r))
                s = np.shape(V)[1]
            else:
                init_U = False
                if not precond: 
                    U = np.zeros((self.d,r))
                else:
                    U = np.zeros((self.d_red,r))
                V = np.random.randn(self.m,s) 
                V,_,_ = np.linalg.svd(V, full_matrices=False)

            iter_U.append(U)
            iter_V.append(V)
            
            # Alternating SVD #####################################################
            i = 0
            while i < max_iter:     
                print(f'AlternEIG Iteration {i} ----------------')
                if init_U:
                    HY = self.compute_HY_U(U, precond = precond, full_grad=full_grad)
                    V, _ = utils.compute_ev_ew(HY, s)
                    init_U=False
                else:
                    HX = self.compute_HX_V(V, precond = precond, full_grad=full_grad)
                    U, _ = utils.compute_ev_ew(HX, r)
                    init_U=True
                i += 1
                iter_U.append(U)
                iter_V.append(V)
        
            if save:
                with open(f'{filename}', 'wb') as f:  
                    print(f'Saving coupled results in {filename}')
                    if full_grad:
                        pickle.dump([HX,HY,iter_U,iter_V], f)
                    else:
                        _, ew_HX = utils.compute_ev_ew(HX,r)
                        _, ew_HY = utils.compute_ev_ew(HY, s)
                        pickle.dump([ew_HX,ew_HY,iter_U,iter_V], f)
                    
                
        return HX, HY, iter_U, iter_V
    

###################################
#%% Initiate DR Instance
###################################

def init_model(model_name, **kwargs):
    """ Use kwargs to specify model parameters   """
    if model_name == 'conddiff':
        model = cd.init_conddiff(**kwargs)
    elif model_name[:7] == 'advdiff':
        model = ad.init_advdiff(**kwargs)
    elif model_name[:7] == 'burgers':
        model = bg.init_burgers(**kwargs)
    elif model_name == 'blockdiff':
        model = bd.init_blockdiff(**kwargs)
    return model



def init_dr(model_name, M=1000, full_grad=True, load=True, save =True, cache_tag='', **kwargs):
    model = init_model(model_name, **kwargs)
    bayes = bip.init_bayes(model,cache_tag=cache_tag, **kwargs)
    problem = DR(bayes=bayes,M=M,full_grad=full_grad,load=load, save=save,cache_tag=cache_tag)
    return problem